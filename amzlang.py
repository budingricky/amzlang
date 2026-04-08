import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib import error, request


def load_env_file(file_path: str = ".env") -> None:
    if not os.path.exists(file_path):
        return
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        return


@dataclass
class RunResult:
    success: bool
    generated_code: str
    output: str
    error: str
    attempts: int
    logs: list[str]
    used_ai: bool
    ai_attempted: bool


@dataclass
class TranslationReport:
    code: str
    used_ai: bool
    ai_attempted: bool
    mode: str
    message: str


class AIRequiredError(RuntimeError):
    def __init__(self, message: str, ai_attempted: bool) -> None:
        super().__init__(message)
        self.ai_attempted = ai_attempted


class AITranslator:
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        load_env_file()
        selected_provider = provider or os.getenv("LLM_PROVIDER", "deepseek")
        self.provider = selected_provider.strip().lower()
        if self.provider == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
            self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        else:
            self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", os.getenv("OPENAI_API_KEY", ""))
            self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
            self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    def translate(
        self,
        source_text: str,
        previous_error: Optional[str] = None,
        previous_code: Optional[str] = None,
        require_ai: bool = False,
    ) -> TranslationReport:
        api_output, api_message, ai_attempted = self._translate_via_api(source_text, previous_error, previous_code)
        if api_output:
            return TranslationReport(
                code=api_output,
                used_ai=True,
                ai_attempted=ai_attempted,
                mode="ai",
                message=api_message,
            )
        if require_ai:
            detail = api_message or "AI translation unavailable"
            raise AIRequiredError(f"AI translation required but failed: {detail}", ai_attempted=ai_attempted)
        local_code, local_mode = self._translate_locally(source_text, previous_error, previous_code)
        return TranslationReport(
            code=local_code,
            used_ai=False,
            ai_attempted=ai_attempted,
            mode=local_mode,
            message=api_message,
        )

    def _translate_via_api(
        self,
        source_text: str,
        previous_error: Optional[str],
        previous_code: Optional[str],
    ) -> tuple[Optional[str], str, bool]:
        if not self.api_key:
            return None, "API key is missing", False

        system_prompt = (
            "You are a compiler backend. Convert any user input into valid executable Python code only. "
            "Return pure Python source with no markdown and no explanations. "
            "If there is a previous execution error, fix it and return corrected code."
        )

        user_payload = {
            "source_text": source_text,
            "previous_error": previous_error or "",
            "previous_code": previous_code or "",
        }

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "temperature": 0.2,
        }
        data = json.dumps(body).encode("utf-8")
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = request.Request(url=url, data=data, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=25) as resp:
                response_text = resp.read().decode("utf-8")
            payload = json.loads(response_text)
            content = payload["choices"][0]["message"]["content"]
            if isinstance(content, list):
                joined = []
                for item in content:
                    if isinstance(item, dict):
                        joined.append(str(item.get("text", "")))
                    else:
                        joined.append(str(item))
                content = "".join(joined)
            return self._strip_markdown_fence(str(content)), "AI request succeeded", True
        except error.HTTPError as e:
            return None, f"HTTPError {e.code}: {e.reason}", True
        except error.URLError as e:
            return None, f"URLError: {e.reason}", True
        except TimeoutError:
            return None, "TimeoutError", True
        except (KeyError, IndexError, json.JSONDecodeError):
            return None, "Malformed API response", True

    def _translate_locally(
        self,
        source_text: str,
        previous_error: Optional[str],
        previous_code: Optional[str],
    ) -> tuple[str, str]:
        text = source_text.strip()

        if previous_error and previous_code:
            fixed = self._simple_repair(previous_code, previous_error)
            if fixed != previous_code:
                return fixed, "local_repair"

        js_log = self._from_js_console_log(text)
        if js_log:
            return js_log, "local_js"

        if self._is_python(text):
            return text, "local_python"

        c_printf = self._from_c_printf(text)
        if c_printf:
            return c_printf, "local_c"

        natural = self._from_natural_language(text)
        if natural:
            return natural, "local_nl"

        expr = self._from_expression(text)
        if expr:
            return expr, "local_expr"

        return f'print({json.dumps(text, ensure_ascii=False)})', "local_default"

    def _strip_markdown_fence(self, content: str) -> str:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
        return cleaned.strip()

    def _is_python(self, text: str) -> bool:
        if not text:
            return False
        try:
            ast.parse(text)
            return True
        except SyntaxError:
            return False

    def _from_js_console_log(self, text: str) -> Optional[str]:
        pattern = re.compile(r"console\.log\((.+?)\)\s*;?", re.IGNORECASE)
        matches = pattern.findall(text)
        if not matches:
            return None
        lines = [f"print({m.strip()})" for m in matches]
        return "\n".join(lines)

    def _from_c_printf(self, text: str) -> Optional[str]:
        pattern = re.compile(r'printf\(\s*"([^"]*)"(?:\s*,\s*([^)]+))?\s*\)\s*;?', re.IGNORECASE)
        match = pattern.search(text)
        if not match:
            return None
        fmt = match.group(1)
        args = match.group(2)
        if not args:
            return f"print({json.dumps(fmt, ensure_ascii=False)})"
        return f'print("{fmt}" % ({args.strip()}))'

    def _from_natural_language(self, text: str) -> Optional[str]:
        zh = re.search(r"(打印|输出)\s*[：: ]?\s*(.+)", text)
        if zh:
            return f'print({json.dumps(zh.group(2).strip(), ensure_ascii=False)})'

        en = re.search(r"(print|say|output)\s*[: ]\s*(.+)", text, re.IGNORECASE)
        if en:
            return f'print({json.dumps(en.group(2).strip(), ensure_ascii=False)})'
        return None

    def _from_expression(self, text: str) -> Optional[str]:
        expr = text.replace(" ", "")
        if re.fullmatch(r"[0-9\.\+\-\*\/\(\)]+", expr):
            return f"print({expr})"
        return None

    def _simple_repair(self, code: str, err: str) -> str:
        fixed = code
        if ("AttributeError" in err and "has no attribute 'log'" in err) or "NameError: name 'console'" in err:
            converted = self._from_js_console_log(code)
            if converted:
                return converted
        if "NameError" in err:
            name_match = re.search(r"name '([^']+)' is not defined", err)
            if name_match:
                name = name_match.group(1)
                if f"{name}.log(" in fixed:
                    fixed = fixed.replace(f"{name}.log(", "console.log(")
                    converted = self._from_js_console_log(fixed)
                    if converted:
                        return converted
                fixed = f"{name} = None\n{fixed}"
        if "SyntaxError" in err and not fixed.endswith("\n"):
            fixed += "\n"
        return fixed


class AMZLanguage:
    def __init__(
        self,
        max_attempts: int = 5,
        timeout_sec: int = 8,
        require_ai: bool = False,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.max_attempts = max_attempts
        self.timeout_sec = timeout_sec
        self.require_ai = require_ai
        self.translator = AITranslator(provider=provider, model=model, base_url=base_url, api_key=api_key)

    def run(self, source_text: str) -> RunResult:
        previous_error = None
        previous_code = None
        last_code = ""
        last_error = "Unknown failure"
        logs: list[str] = [
            f"编译器: provider={self.translator.provider}, model={self.translator.model}",
            f"配置: max_attempts={self.max_attempts}, timeout={self.timeout_sec}s, require_ai={self.require_ai}",
        ]
        used_ai = False
        ai_attempted = False

        for attempt in range(1, self.max_attempts + 1):
            logs.append(f"[Attempt {attempt}] 开始转译")
            try:
                translation = self.translator.translate(
                    source_text,
                    previous_error,
                    previous_code,
                    require_ai=self.require_ai,
                )
            except AIRequiredError as e:
                last_error = str(e)
                ai_attempted = ai_attempted or e.ai_attempted
                logs.append(f"[Attempt {attempt}] 转译失败: {last_error}")
                return RunResult(
                    success=False,
                    generated_code=last_code,
                    output="",
                    error=last_error,
                    attempts=attempt,
                    logs=logs,
                    used_ai=used_ai,
                    ai_attempted=ai_attempted,
                )

            code = translation.code
            last_code = code
            used_ai = used_ai or translation.used_ai
            ai_attempted = ai_attempted or translation.ai_attempted
            logs.append(
                f"[Attempt {attempt}] 转译模式={translation.mode}, used_ai={translation.used_ai}, ai_attempted={translation.ai_attempted}"
            )
            logs.append(f"[Attempt {attempt}] 转译详情={translation.message}")
            logs.append(f"[Attempt {attempt}] 转译结果:\n{code}")

            syntax_error = self._syntax_check(code)
            if syntax_error:
                logs.append(f"[Attempt {attempt}] 语法检查失败: {syntax_error}")
                previous_error = syntax_error
                previous_code = code
                last_error = syntax_error
                continue
            logs.append(f"[Attempt {attempt}] 语法检查通过")

            ok, out, err, exec_cmd = self._execute_python(code)
            logs.append(f"[Attempt {attempt}] 执行命令: {exec_cmd}")
            if ok:
                logs.append(f"[Attempt {attempt}] 执行成功")
                return RunResult(
                    success=True,
                    generated_code=code,
                    output=out,
                    error="",
                    attempts=attempt,
                    logs=logs,
                    used_ai=used_ai,
                    ai_attempted=ai_attempted,
                )

            logs.append(f"[Attempt {attempt}] 执行失败: {err.strip()}")
            previous_error = err
            previous_code = code
            last_error = err

        logs.append("编译结束: 重试次数耗尽")
        return RunResult(
            success=False,
            generated_code=last_code,
            output="",
            error=last_error,
            attempts=self.max_attempts,
            logs=logs,
            used_ai=used_ai,
            ai_attempted=ai_attempted,
        )

    def _syntax_check(self, code: str) -> Optional[str]:
        try:
            compile(code, "<amzlang>", "exec")
            return None
        except SyntaxError as e:
            return f"SyntaxError: {e}"

    def _execute_python(self, code: str) -> tuple[bool, str, str, str]:
        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".py", encoding="utf-8") as tmp:
                tmp.write(code)
                temp_path = tmp.name

            exec_cmd = f'"{sys.executable}" "{temp_path}"'
            proc = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                timeout=self.timeout_sec,
            )
            if proc.returncode == 0:
                return True, proc.stdout, "", exec_cmd
            return False, proc.stdout, proc.stderr or f"Process exited with code {proc.returncode}", exec_cmd
        except subprocess.TimeoutExpired:
            return False, "", f"RuntimeError: timed out after {self.timeout_sec} seconds", "python <tempfile>"
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass


def read_source(file_path: Optional[str], inline_text: Optional[str], require_amz: bool = False) -> str:
    if inline_text is not None:
        return inline_text
    if file_path:
        path = Path(file_path)
        if require_amz and path.suffix.lower() != ".amz":
            raise ValueError("Source file must use .amz extension")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return sys.stdin.read()


def create_compiler_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="amz", description="AMZ compiler")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--max-attempts", type=int, default=5)
    common.add_argument("--timeout", type=int, default=8)
    common.add_argument("--provider", choices=["deepseek", "openai"], help="LLM provider")
    common.add_argument("--model", type=str, help="Override model name")
    common.add_argument("--base-url", type=str, help="Override API base URL")
    common.add_argument("--api-key", type=str, help="Override API key")
    common.add_argument("--require-ai", action="store_true", help="Fail when AI translation is unavailable")
    common.add_argument("--no-logs", action="store_true", help="Hide compile logs")
    common.add_argument("--show-code", action="store_true")

    run_parser = subparsers.add_parser("run", parents=[common], help="Compile and run source")
    run_parser.add_argument("source", nargs="?", help=".amz source file")
    run_parser.add_argument("--text", type=str, help="Inline source text")

    build_parser = subparsers.add_parser("build", parents=[common], help="Compile source into CLI command")
    build_parser.add_argument("source", help=".amz source file")
    build_parser.add_argument("-o", "--output", required=True, help="Output command name")
    build_parser.add_argument("--emit", choices=["cmd", "py", "both"], default="both")

    install_parser = subparsers.add_parser("install", help="Install amz command wrapper in current directory")
    install_parser.add_argument("--name", default="amz", help="Wrapper command name")

    native_parser = subparsers.add_parser("native", help="Build native amz executable")
    native_parser.add_argument("--name", default="amz", help="Native executable name")
    native_parser.add_argument("--dist", default="dist-native", help="Output directory for executable")
    native_parser.add_argument("--onefile", action="store_true", help="Build onefile executable")
    native_parser.add_argument("--windowed", action="store_true", help="Build windowed executable")
    return parser


def create_runner_from_args(args: argparse.Namespace) -> AMZLanguage:
    return AMZLanguage(
        max_attempts=args.max_attempts,
        timeout_sec=args.timeout,
        require_ai=args.require_ai,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    )


def print_generated_code(code: str) -> None:
    print("=== GENERATED PYTHON ===")
    print(code)
    print("=== END GENERATED PYTHON ===")


def print_compile_logs(logs: list[str], used_ai: bool, ai_attempted: bool) -> None:
    print("=== COMPILE LOG START ===")
    print(f"AI调用状态: used_ai={used_ai}, ai_attempted={ai_attempted}")
    for line in logs:
        print(line)
    print("=== COMPILE LOG END ===")


def handle_run_command(args: argparse.Namespace) -> int:
    try:
        source = read_source(args.source, args.text, require_amz=bool(args.source))
    except (OSError, ValueError) as e:
        print(str(e), file=sys.stderr)
        return 2

    runner = create_runner_from_args(args)
    result = runner.run(source)

    if not args.no_logs:
        print_compile_logs(result.logs, result.used_ai, result.ai_attempted)
    if args.show_code:
        print_generated_code(result.generated_code)

    if result.success:
        if result.output:
            print(result.output, end="" if result.output.endswith("\n") else "\n")
        return 0

    print(f"Compile/Run failed after {result.attempts} attempts.", file=sys.stderr)
    print(result.error, file=sys.stderr)
    return 1


def build_cmd_wrapper(py_file_name: str) -> str:
    return (
        "@echo off\n"
        "setlocal\n"
        "where py >nul 2>nul\n"
        "if %errorlevel%==0 (\n"
        f'  py -3 "%~dp0{py_file_name}" %*\n'
        ") else (\n"
        f'  python "%~dp0{py_file_name}" %*\n'
        ")\n"
    )


def handle_build_command(args: argparse.Namespace) -> int:
    try:
        source = read_source(args.source, None, require_amz=True)
    except (OSError, ValueError) as e:
        print(str(e), file=sys.stderr)
        return 2

    runner = create_runner_from_args(args)
    result = runner.run(source)
    if not args.no_logs:
        print_compile_logs(result.logs, result.used_ai, result.ai_attempted)
    if args.show_code:
        print_generated_code(result.generated_code)
    if not result.success:
        print(f"Compile failed after {result.attempts} attempts.", file=sys.stderr)
        print(result.error, file=sys.stderr)
        return 1

    output_path = Path(args.output)
    py_path = output_path.with_suffix(".py")
    cmd_path = output_path.with_suffix(".cmd")
    py_name = py_path.name

    try:
        if args.emit in {"py", "both"}:
            py_path.write_text(result.generated_code + ("\n" if not result.generated_code.endswith("\n") else ""), encoding="utf-8")
        if args.emit in {"cmd", "both"}:
            cmd_path.write_text(build_cmd_wrapper(py_name), encoding="utf-8")
    except OSError as e:
        print(f"Write output failed: {e}", file=sys.stderr)
        return 2

    if args.emit == "py":
        print(f"Compiled python: {py_path}")
    elif args.emit == "cmd":
        print(f"Compiled command wrapper: {cmd_path}")
    else:
        print(f"Compiled command: {cmd_path.stem}")
        print(f"Files: {py_path}, {cmd_path}")
    return 0


def handle_install_command(args: argparse.Namespace) -> int:
    wrapper_name = Path(args.name).stem
    wrapper_path = Path.cwd() / f"{wrapper_name}.cmd"
    target_script = Path(__file__).name
    script = (
        "@echo off\n"
        "setlocal\n"
        f'set SCRIPT=%~dp0{target_script}\n'
        "where py >nul 2>nul\n"
        "if %errorlevel%==0 (\n"
        '  py -3 "%SCRIPT%" %*\n'
        ") else (\n"
        '  python "%SCRIPT%" %*\n'
        ")\n"
    )
    try:
        wrapper_path.write_text(script, encoding="utf-8")
    except OSError as e:
        print(f"Install failed: {e}", file=sys.stderr)
        return 2
    print(f"Installed CLI wrapper: {wrapper_path}")
    print(f"Usage: {wrapper_name} run your_file.amz")
    return 0


def handle_native_command(args: argparse.Namespace) -> int:
    script_path = Path(__file__).resolve()
    dist_dir = Path(args.dist).resolve()
    build_dir = dist_dir / "build"
    work_dir = dist_dir / "work"
    spec_dir = dist_dir / "spec"
    dist_out = dist_dir / "bin"
    exe_name = Path(args.name).stem
    pyinstaller_args = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name",
        exe_name,
        "--distpath",
        str(dist_out),
        "--workpath",
        str(work_dir),
        "--specpath",
        str(spec_dir),
    ]
    if args.onefile:
        pyinstaller_args.append("--onefile")
    if args.windowed:
        pyinstaller_args.append("--windowed")
    pyinstaller_args.append(str(script_path))

    try:
        check_cmd = [sys.executable, "-m", "PyInstaller", "--version"]
        check = subprocess.run(check_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if check.returncode != 0:
            print("PyInstaller is not installed. Run: pip install pyinstaller", file=sys.stderr)
            return 2
    except OSError as e:
        print(f"Failed to check PyInstaller: {e}", file=sys.stderr)
        return 2

    try:
        build_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        spec_dir.mkdir(parents=True, exist_ok=True)
        dist_out.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Prepare output directories failed: {e}", file=sys.stderr)
        return 2

    print("Building native amz CLI...")
    proc = subprocess.run(pyinstaller_args, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print("Native build failed.", file=sys.stderr)
        return 1

    suffix = ".exe" if os.name == "nt" else ""
    built_exe = dist_out / f"{exe_name}{suffix}"
    if not built_exe.exists():
        print(f"Native build output not found: {built_exe}", file=sys.stderr)
        return 2

    target_exe = Path.cwd() / built_exe.name
    try:
        shutil.copy2(built_exe, target_exe)
    except OSError as e:
        print(f"Copy executable failed: {e}", file=sys.stderr)
        return 2

    print(f"Native CLI built: {target_exe}")
    print(f"Usage: .\\{target_exe.name} -h")
    return 0


def main() -> int:
    parser = create_compiler_parser()
    args = parser.parse_args()
    if args.command == "run":
        return handle_run_command(args)
    if args.command == "build":
        return handle_build_command(args)
    if args.command == "native":
        return handle_native_command(args)
    return handle_install_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
