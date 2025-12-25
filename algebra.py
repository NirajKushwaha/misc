from .utils import *

class Halley2D:
    """
    Solve a system of two nonlinear equations using Halley's 2nd order method.    
    """

    def __init__(self, f1, f2, h=1e-6):
        """
        f1, f2 : callable
            Functions f1(x,y), f2(x,y).
            ex-> def f1(x,y): return x**2 + y**2 - 1
        h : float
            Step size for numerical differentiation
        """
        self.f1 = f1
        self.f2 = f2
        self.h = h

    def F(self, x, y):
        return np.array([self.f1(x, y), self.f2(x, y)])

    def jacobian(self, x, y):
        h = self.h

        df1_dx = (self.f1(x + h, y) - self.f1(x - h, y)) / (2*h)
        df1_dy = (self.f1(x, y + h) - self.f1(x, y - h)) / (2*h)

        df2_dx = (self.f2(x + h, y) - self.f2(x - h, y)) / (2*h)
        df2_dy = (self.f2(x, y + h) - self.f2(x, y - h)) / (2*h)

        return np.array([
            [df1_dx, df1_dy],
            [df2_dx, df2_dy]
        ])

    def hessian(self, f, x, y):
        h = self.h

        dxx = (f(x + h, y) - 2*f(x, y) + f(x - h, y)) / h**2
        dyy = (f(x, y + h) - 2*f(x, y) + f(x, y - h)) / h**2
        dxy = (
            f(x + h, y + h)
            - f(x + h, y - h)
            - f(x - h, y + h)
            + f(x - h, y - h)
        ) / (4*h**2)

        return np.array([
            [dxx, dxy],
            [dxy, dyy]
        ])

    def solve(self, x0, y0, tol=1e-10, max_iter=50):
            x, y = x0, y0

            for _ in range(max_iter):
                F = self.F(x, y)
                if np.linalg.norm(F) < tol:
                    # ---- STABILITY CHECK ----
                    J = self.jacobian(x, y)

                    eigvals = np.linalg.eigvals(J)
                    cond_number = np.linalg.cond(J)

                    stable = (
                        np.all(np.abs(eigvals) > 1e-8) and
                        cond_number < 1e8
                    )

                    if not stable:
                        return np.nan, np.nan ## Halley method converged but the solution is unstable
                    # --------------------------------

                    return x, y ## Halley method converged with stable solution


                J = self.jacobian(x, y)
                J_inv = np.linalg.inv(J)

                delta = J_inv @ F

                H1 = self.hessian(self.f1, x, y)
                H2 = self.hessian(self.f2, x, y)

                H_delta = np.array([
                    H1 @ delta,
                    H2 @ delta
                ])

                correction = np.eye(2) - 0.5 * J_inv @ H_delta
                step = np.linalg.inv(correction) @ delta

                x -= step[0]
                y -= step[1]

            return np.nan, np.nan ## Halley method did not converge