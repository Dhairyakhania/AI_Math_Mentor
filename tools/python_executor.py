"""
Safe Python code executor for mathematical computations.
Provides a sandboxed environment for running Python code.
"""

from agno.tools import tool
import ast
import sys
import io
import traceback
from typing import Optional
from contextlib import redirect_stdout, redirect_stderr
import signal
import math
import numpy as np
import sympy as sp


# Safe builtins allowed in execution
SAFE_BUILTINS = {
    'abs': abs,
    'all': all,
    'any': any,
    'bin': bin,
    'bool': bool,
    'chr': chr,
    'dict': dict,
    'divmod': divmod,
    'enumerate': enumerate,
    'filter': filter,
    'float': float,
    'format': format,
    'frozenset': frozenset,
    'hex': hex,
    'int': int,
    'isinstance': isinstance,
    'len': len,
    'list': list,
    'map': map,
    'max': max,
    'min': min,
    'oct': oct,
    'ord': ord,
    'pow': pow,
    'print': print,
    'range': range,
    'repr': repr,
    'reversed': reversed,
    'round': round,
    'set': set,
    'sorted': sorted,
    'str': str,
    'sum': sum,
    'tuple': tuple,
    'type': type,
    'zip': zip,
    'True': True,
    'False': False,
    'None': None,
}

# Safe modules allowed in execution
SAFE_MODULES = {
    'math': math,
    'np': np,
    'numpy': np,
    'sp': sp,
    'sympy': sp,
}

# Dangerous patterns to block
DANGEROUS_PATTERNS = [
    'import os',
    'import sys',
    'import subprocess',
    'import socket',
    'import requests',
    '__import__',
    'eval(',
    'exec(',
    'compile(',
    'open(',
    'file(',
    'input(',
    'raw_input(',
    '__builtins__',
    '__class__',
    '__bases__',
    '__subclasses__',
    '__mro__',
    '__globals__',
    '__code__',
    'getattr',
    'setattr',
    'delattr',
    'globals(',
    'locals(',
    'vars(',
    'dir(',
]


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out")


class SafePythonExecutor:
    """
    A sandboxed Python executor for mathematical computations.
    """
    
    def __init__(self, timeout: int = 10, max_output_length: int = 10000):
        self.timeout = timeout
        self.max_output_length = max_output_length
    
    def _check_code_safety(self, code: str) -> tuple[bool, str]:
        """
        Check if code is safe to execute.
        Returns (is_safe, error_message)
        """
        # Check for dangerous patterns
        code_lower = code.lower()
        for pattern in DANGEROUS_PATTERNS:
            if pattern.lower() in code_lower:
                return False, f"Dangerous pattern detected: {pattern}"
        
        # Try to parse the code to check syntax
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        # Walk the AST to check for dangerous operations
        for node in ast.walk(tree):
            # Block certain node types
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in ['math', 'numpy', 'sympy', 'np', 'sp']:
                        return False, f"Import not allowed: {alias.name}"
            
            elif isinstance(node, ast.ImportFrom):
                if node.module not in ['math', 'numpy', 'sympy']:
                    return False, f"Import from not allowed: {node.module}"
            
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'compile', 'open', '__import__']:
                        return False, f"Function not allowed: {node.func.id}"
        
        return True, ""
    
    def _create_safe_globals(self) -> dict:
        """Create a safe globals dictionary for execution"""
        safe_globals = {
            '__builtins__': SAFE_BUILTINS,
            '__name__': '__main__',
        }
        safe_globals.update(SAFE_MODULES)
        
        # Add common math functions directly
        safe_globals.update({
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'sqrt': math.sqrt,
            'log': math.log,
            'log10': math.log10,
            'exp': math.exp,
            'pi': math.pi,
            'e': math.e,
            'factorial': math.factorial,
            'gcd': math.gcd,
            'floor': math.floor,
            'ceil': math.ceil,
            'radians': math.radians,
            'degrees': math.degrees,
        })
        
        # Add sympy symbols
        safe_globals.update({
            'Symbol': sp.Symbol,
            'symbols': sp.symbols,
            'solve': sp.solve,
            'simplify': sp.simplify,
            'expand': sp.expand,
            'factor': sp.factor,
            'diff': sp.diff,
            'integrate': sp.integrate,
            'limit': sp.limit,
            'Matrix': sp.Matrix,
            'Rational': sp.Rational,
            'oo': sp.oo,
            'I': sp.I,
        })
        
        return safe_globals
    
    def execute(self, code: str) -> dict:
        """
        Execute Python code in a sandboxed environment.
        
        Returns:
            dict with keys:
                - success: bool
                - output: str (stdout)
                - error: str (if any)
                - result: any (last expression result)
        """
        # Check code safety
        is_safe, error_msg = self._check_code_safety(code)
        if not is_safe:
            return {
                "success": False,
                "output": "",
                "error": f"Security check failed: {error_msg}",
                "result": None
            }
        
        # Prepare execution environment
        safe_globals = self._create_safe_globals()
        safe_locals = {}
        
        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        result = None
        error = None
        
        try:
            # Set timeout (Unix only)
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout)
            
            # Execute code
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Try to get a result from the last expression
                try:
                    tree = ast.parse(code)
                    
                    # If last statement is an expression, capture its value
                    if tree.body and isinstance(tree.body[-1], ast.Expr):
                        # Execute all but last statement
                        if len(tree.body) > 1:
                            exec_code = ast.Module(body=tree.body[:-1], type_ignores=[])
                            exec(compile(exec_code, '<string>', 'exec'), safe_globals, safe_locals)
                        
                        # Evaluate last expression
                        last_expr = ast.Expression(body=tree.body[-1].value)
                        result = eval(compile(last_expr, '<string>', 'eval'), safe_globals, safe_locals)
                    else:
                        exec(code, safe_globals, safe_locals)
                        
                except Exception as e:
                    error = str(e)
            
            # Cancel timeout
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                
        except TimeoutError:
            error = f"Execution timed out after {self.timeout} seconds"
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}"
        
        # Get outputs
        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()
        
        # Truncate if too long
        if len(stdout_output) > self.max_output_length:
            stdout_output = stdout_output[:self.max_output_length] + "\n... (output truncated)"
        
        # Combine outputs
        combined_output = stdout_output
        if stderr_output:
            combined_output += f"\nStderr: {stderr_output}"
        
        return {
            "success": error is None,
            "output": combined_output,
            "error": error,
            "result": result
        }


# Global executor instance
_executor = SafePythonExecutor()


# @tool
# def execute_python(code: str) -> str:
#     """
#     Execute Python code for mathematical computations.
    
#     The code runs in a sandboxed environment with access to:
#     - math module (sin, cos, sqrt, log, etc.)
#     - numpy (as np)
#     - sympy (as sp) for symbolic math
    
#     Args:
#         code: Python code to execute. Can use print() for output.
    
#     Returns:
#         The output of the code execution, including any printed output
#         and the result of the last expression.
    
#     Examples:
#         - "2 + 2"  → "4"
#         - "import math; math.sqrt(16)"  → "4.0"
#         - "x = sp.Symbol('x'); sp.solve(x**2 - 4, x)"  → "[-2, 2]"
#     """
#     result = _executor.execute(code)
    
#     if not result["success"]:
#         return f"Error: {result['error']}"
    
#     output_parts = []
    
#     if result["output"]:
#         output_parts.append(result["output"])
    
#     if result["result"] is not None:
#         output_parts.append(f"Result: {result['result']}")
    
#     return "\n".join(output_parts) if output_parts else "Code executed successfully (no output)"

@tool
def execute_python(code: str) -> str:
    """
    Execute Python code safely.
    Returns ONLY the final result or error.
    """
    result = _executor.execute(code)

    if not result["success"]:
        return f"Error: {result['error']}"

    if result["result"] is not None:
        return str(result["result"])

    return result["output"] or "OK"

@tool
def run_math_computation(expression: str, variables: dict = None) -> str:
    """
    Run a mathematical computation with optional variable substitution.
    
    Args:
        expression: Mathematical expression to evaluate
        variables: Optional dictionary of variable values, e.g., {"x": 5, "y": 3}
    
    Returns:
        The computed result
    
    Examples:
        - expression="x**2 + 2*x + 1", variables={"x": 3}  → "16"
        - expression="sin(pi/2)"  → "1.0"
    """
    code_parts = []
    
    # Define variables
    if variables:
        for var_name, var_value in variables.items():
            code_parts.append(f"{var_name} = {var_value}")
    
    # Add the expression
    code_parts.append(expression)
    
    code = "\n".join(code_parts)
    result = _executor.execute(code)
    
    if not result["success"]:
        return f"Error: {result['error']}"
    
    if result["result"] is not None:
        return str(result["result"])
    
    return result["output"] or "No result"


@tool
def symbolic_solve(equation: str, variable: str = "x", domain: str = "real") -> str:
    """
    Solve an equation symbolically using SymPy.
    
    Args:
        equation: Equation to solve. Use "==" for equality or just the left side if equals 0.
        variable: Variable to solve for (default: x)
        domain: Domain for solutions - "real", "complex", or "positive" (default: real)
    
    Returns:
        Solutions to the equation
    
    Examples:
        - equation="x**2 - 4", variable="x"  → "[-2, 2]"
        - equation="x**2 + 1", variable="x", domain="complex"  → "[-I, I]"
    """
    domain_code = ""
    if domain == "real":
        domain_code = f"{variable} = sp.Symbol('{variable}', real=True)"
    elif domain == "positive":
        domain_code = f"{variable} = sp.Symbol('{variable}', positive=True)"
    else:
        domain_code = f"{variable} = sp.Symbol('{variable}')"
    
    # Handle equation with ==
    if "==" in equation:
        left, right = equation.split("==")
        equation = f"({left}) - ({right})"
    
    code = f"""
{domain_code}
solutions = sp.solve({equation}, {variable})
solutions
"""
    
    result = _executor.execute(code)
    
    if not result["success"]:
        return f"Error: {result['error']}"
    
    if result["result"] is not None:
        return f"Solutions: {result['result']}"
    
    return "No solutions found"


@tool
def compute_derivative(expression: str, variable: str = "x", order: int = 1) -> str:
    """
    Compute the derivative of an expression.
    
    Args:
        expression: Mathematical expression to differentiate
        variable: Variable to differentiate with respect to (default: x)
        order: Order of derivative (default: 1)
    
    Returns:
        The derivative
    """
    code = f"""
{variable} = sp.Symbol('{variable}')
expr = {expression}
derivative = sp.diff(expr, {variable}, {order})
sp.simplify(derivative)
"""
    
    result = _executor.execute(code)
    
    if not result["success"]:
        return f"Error: {result['error']}"
    
    if result["result"] is not None:
        return f"d{''.join(['/d'+variable]*order)}({expression}) = {result['result']}"
    
    return "Could not compute derivative"


@tool
def compute_integral(expression: str, variable: str = "x", lower_bound: str = None, upper_bound: str = None) -> str:
    """
    Compute the integral of an expression.
    
    Args:
        expression: Mathematical expression to integrate
        variable: Variable to integrate with respect to (default: x)
        lower_bound: Lower bound for definite integral (optional)
        upper_bound: Upper bound for definite integral (optional)
    
    Returns:
        The integral (with + C for indefinite integrals)
    """
    code = f"{variable} = sp.Symbol('{variable}')\n"
    code += f"expr = {expression}\n"
    
    if lower_bound is not None and upper_bound is not None:
        code += f"integral = sp.integrate(expr, ({variable}, {lower_bound}, {upper_bound}))\n"
        code += "sp.simplify(integral)"
    else:
        code += f"integral = sp.integrate(expr, {variable})\n"
        code += "sp.simplify(integral)"
    
    result = _executor.execute(code)
    
    if not result["success"]:
        return f"Error: {result['error']}"
    
    if result["result"] is not None:
        if lower_bound is None:
            return f"∫({expression})d{variable} = {result['result']} + C"
        else:
            return f"∫[{lower_bound},{upper_bound}]({expression})d{variable} = {result['result']}"
    
    return "Could not compute integral"


@tool
def matrix_operation(operation: str, matrix_a: str, matrix_b: str = None) -> str:
    """
    Perform matrix operations.
    
    Args:
        operation: One of "determinant", "inverse", "multiply", "add", "eigenvalues", "transpose"
        matrix_a: First matrix as string, e.g., "[[1,2],[3,4]]"
        matrix_b: Second matrix for binary operations (optional)
    
    Returns:
        Result of the matrix operation
    """
    code = f"A = sp.Matrix({matrix_a})\n"
    
    if matrix_b:
        code += f"B = sp.Matrix({matrix_b})\n"
    
    if operation == "determinant":
        code += "A.det()"
    elif operation == "inverse":
        code += "A.inv()"
    elif operation == "transpose":
        code += "A.T"
    elif operation == "eigenvalues":
        code += "A.eigenvals()"
    elif operation == "multiply" and matrix_b:
        code += "A * B"
    elif operation == "add" and matrix_b:
        code += "A + B"
    else:
        return f"Unknown operation: {operation}"
    
    result = _executor.execute(code)
    
    if not result["success"]:
        return f"Error: {result['error']}"
    
    if result["result"] is not None:
        return f"{operation.capitalize()} result:\n{result['result']}"
    
    return "Could not perform matrix operation"


@tool
def evaluate_expression(expression: str, precision: int = 10) -> str:
    """
    Evaluate a mathematical expression numerically.
    
    Args:
        expression: Expression to evaluate
        precision: Number of decimal places (default: 10)
    
    Returns:
        Numerical result
    """
    code = f"""
result = {expression}
if hasattr(result, 'evalf'):
    float(result.evalf({precision}))
else:
    round(float(result), {precision})
"""
    
    result = _executor.execute(code)
    
    if not result["success"]:
        return f"Error: {result['error']}"
    
    if result["result"] is not None:
        return str(result["result"])
    
    return "Could not evaluate expression"