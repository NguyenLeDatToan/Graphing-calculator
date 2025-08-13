from __future__ import annotations
from typing import Iterable, List, Optional, Tuple

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

x, y = sp.symbols('x y')


def parse_functions(func_strs: Iterable[str]) -> List[sp.Expr]:
    """Chuyển danh sách chuỗi hàm số thành SymPy expressions.
    Ví dụ: ["x**2", "sin(x)"] → [x**2, sin(x)]
    """
    exprs: List[sp.Expr] = []
    for s in func_strs:
        exprs.append(sp.sympify(s, convert_xor=True))
    return exprs


def plot_functions(
    func_strs: Iterable[str],
    x_range: Tuple[float, float] = (-10, 10),
    shade: Optional[Tuple[str, str]] = None,  # ("above"|"below", func_str)
    zoom: Optional[Tuple[float, float, float, float]] = None,  # (xmin,xmax,ymin,ymax)
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Vẽ một hoặc nhiều hàm số trên cùng hệ trục.

    - shade: (mode, func_str) để tô vùng trên/below đường của func_str.
    - zoom: (xmin, xmax, ymin, ymax) để phóng to/thu nhỏ vùng nhìn.
    """
    exprs = parse_functions(func_strs)
    f_lambdas = [sp.lambdify(x, e, 'numpy') for e in exprs]

    xmin, xmax = x_range
    xs = np.linspace(xmin, xmax, 1000)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    # Vẽ các hàm
    for s, f in zip(func_strs, f_lambdas):
        ys = f(xs)
        ax.plot(xs, ys, label=s)

    # Shade trên/dưới
    if shade is not None:
        mode, shade_func_str = shade
        shade_expr = sp.sympify(shade_func_str, convert_xor=True)
        shade_lambda = sp.lambdify(x, shade_expr, 'numpy')
        ys_shade = shade_lambda(xs)
        if mode.lower() == 'above':
            ax.fill_between(xs, ys_shade, ax.get_ylim()[1], alpha=0.2, color='C0')
        elif mode.lower() == 'below':
            ax.fill_between(xs, ax.get_ylim()[0], ys_shade, alpha=0.2, color='C0')
        else:
            raise ValueError("shade mode must be 'above' or 'below'")

    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # Zoom
    if zoom is not None:
        ax.set_xlim(zoom[0], zoom[1])
        ax.set_ylim(zoom[2], zoom[3])

    return ax


def table_xy(func_str: str, x_values: Iterable[float]) -> List[Tuple[float, float]]:
    """Tạo bảng (x, y) cho một hàm cho trước và danh sách x.
    Trả về danh sách tuple (x, y).
    """
    expr = sp.sympify(func_str, convert_xor=True)
    f = sp.lambdify(x, expr, 'numpy')
    table: List[Tuple[float, float]] = []
    for xv in x_values:
        yv = float(f(xv))
        table.append((float(xv), yv))
    return table


def solve_and_plot_system(
    eq_strs: Iterable[str],
    x_range: Tuple[float, float] = (-10, 10),
    y_range: Tuple[float, float] = (-10, 10),
    ax: Optional[plt.Axes] = None,
) -> Tuple[List[Tuple[float, float]], plt.Axes]:
    """Giải và vẽ hệ phương trình 2 ẩn.

    - eq_strs: dạng ["y - x - 1", "x + y - 3"] hiểu là = 0.
      Hoặc dạng có '=' như "y = x + 1" sẽ được chuyển về vế trái = 0.
    Trả về (danh sách nghiệm (x,y), axes đã vẽ các đường/hàm liên quan).
    """
    equations = []
    for s in eq_strs:
        if '=' in s:
            left, right = s.split('=', 1)
            equations.append(sp.sympify(left) - sp.sympify(right))
        else:
            equations.append(sp.sympify(s))

    sols = sp.solve(equations, (x, y), dict=True)
    pts: List[Tuple[float, float]] = []
    for sol in sols:
        xv = float(sol[x])
        yv = float(sol[y])
        pts.append((xv, yv))

    # Vẽ các đường mức = 0 tương ứng
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    u = np.linspace(x_range[0], x_range[1], 400)
    v = np.linspace(y_range[0], y_range[1], 400)
    U, V = np.meshgrid(u, v)

    for eq in equations:
        f = sp.lambdify((x, y), eq, 'numpy')
        Z = f(U, V)
        ax.contour(U, V, Z, levels=[0], colors=['C1'], linewidths=1.2)

    # Vẽ điểm nghiệm
    if pts:
        Xp, Yp = zip(*pts)
        ax.scatter(Xp, Yp, color='red', zorder=5, label='solutions')

    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    return pts, ax


def solve_quadratic(a: float, b: float, c: float):
    """Giải phương trình bậc hai ax^2 + bx + c = 0, trả về nghiệm (có thể phức)."""
    A, B, C = map(sp.sympify, (a, b, c))
    roots = sp.solve(sp.Eq(A*x**2 + B*x + C, 0), x)
    return roots
