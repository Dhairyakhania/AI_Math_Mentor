"""
Mathematical calculation tools for Agno agents.
"""

from agno.tools import tool
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)


def _clean_expr(expr: str) -> str:
    """Clean expression for parsing"""
    return expr.replace('^', '**').replace('×', '*').replace('÷', '/')


def _get_transforms():
    """Get sympy transformations"""
    return standard_transformations + (implicit_multiplication_application, convert_xor)


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: Math expression like "2 + 3 * 4", "sqrt(16)", "sin(pi/2)"
    
    Returns:
        The calculated result
    """
    try:
        expr = _clean_expr(expression)
        parsed = parse_expr(expr, transformations=_get_transforms())
        result = parsed.evalf()
        
        if result == int(result):
            return str(int(result))
        return str(round(float(result), 10))
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def solve_equation(equation: str, variable: str = "x") -> str:
    """
    Solve an algebraic equation.
    
    Args:
        equation: Equation like "x**2 - 5*x + 6 = 0" or "x**2 - 4"
        variable: Variable to solve for (default: x)
    
    Returns:
        Solutions to the equation
    """
    try:
        var = sp.Symbol(variable)
        equation = _clean_expr(equation)
        
        if '=' in equation:
            parts = equation.split('=')
            left = parse_expr(parts[0].strip(), transformations=_get_transforms())
            right = parse_expr(parts[1].strip(), transformations=_get_transforms())
            expr = left - right
        else:
            expr = parse_expr(equation, transformations=_get_transforms())
        
        solutions = sp.solve(expr, var)
        
        if not solutions:
            return "No solutions found"
        
        sol_strs = [str(sp.simplify(s)) for s in solutions]
        
        if len(sol_strs) == 1:
            return f"{variable} = {sol_strs[0]}"
        return f"{variable} = {', '.join(sol_strs)}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def differentiate(expression: str, variable: str = "x", order: int = 1) -> str:
    """
    Find the derivative of an expression.
    
    Args:
        expression: Expression to differentiate
        variable: Variable (default: x)
        order: Order of derivative (default: 1)
    
    Returns:
        The derivative
    """
    try:
        var = sp.Symbol(variable)
        expr = parse_expr(_clean_expr(expression), transformations=_get_transforms())
        deriv = sp.diff(expr, var, order)
        return f"d/d{variable}({expression}) = {sp.simplify(deriv)}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def integrate(expression: str, variable: str = "x", lower: str = None, upper: str = None) -> str:
    """
    Find the integral of an expression.
    
    Args:
        expression: Expression to integrate
        variable: Variable (default: x)
        lower: Lower bound for definite integral (optional)
        upper: Upper bound for definite integral (optional)
    
    Returns:
        The integral
    """
    try:
        var = sp.Symbol(variable)
        expr = parse_expr(_clean_expr(expression), transformations=_get_transforms())
        
        if lower and upper:
            low = parse_expr(_clean_expr(lower))
            up = parse_expr(_clean_expr(upper))
            result = sp.integrate(expr, (var, low, up))
            return f"∫[{lower},{upper}]({expression})d{variable} = {sp.simplify(result)}"
        else:
            result = sp.integrate(expr, var)
            return f"∫({expression})d{variable} = {sp.simplify(result)} + C"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def simplify_expression(expression: str) -> str:
    """
    Simplify a mathematical expression.
    
    Args:
        expression: Expression to simplify
    
    Returns:
        Simplified expression
    """
    try:
        expr = parse_expr(_clean_expr(expression), transformations=_get_transforms())
        return f"Simplified: {sp.simplify(expr)}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def factor_expression(expression: str) -> str:
    """
    Factor a polynomial expression.
    
    Args:
        expression: Expression to factor
    
    Returns:
        Factored form
    """
    try:
        expr = parse_expr(_clean_expr(expression), transformations=_get_transforms())
        return f"Factored: {sp.factor(expr)}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def evaluate_limit(expression: str, variable: str = "x", point: str = "0") -> str:
    """
    Evaluate the limit of an expression.
    
    Args:
        expression: Expression to find limit of
        variable: Variable approaching point
        point: Point approached (use "oo" for infinity)
    
    Returns:
        The limit value
    """
    try:
        var = sp.Symbol(variable)
        expr = parse_expr(_clean_expr(expression), transformations=_get_transforms())
        
        if point.lower() in ["oo", "inf", "infinity"]:
            pt = sp.oo
        elif point.lower() in ["-oo", "-inf"]:
            pt = -sp.oo
        else:
            pt = parse_expr(_clean_expr(point))
        
        result = sp.limit(expr, var, pt)
        return f"lim({variable}→{point}) {expression} = {result}"
    except Exception as e:
        return f"Error: {str(e)}"