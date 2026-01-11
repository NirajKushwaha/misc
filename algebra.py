from .utils import *

class Halley2D:
    """
    Solve a system of two nonlinear equations using Halley's 2nd order method
    starting from an initial guess (x0, y0).
    """

    def __init__(self, f1, f2, args=(), h=1e-6, iprint=False):
        """
        Parameters
        ----------
        f1, f2 : callable
            Functions f1(x, y, *args), f2(x, y, *args)
        h : float
            Step size for numerical differentiation
        args : tuple, ()
            Extra positional parameters passed to f1 and f2.
            Pass args as a tuple, even if there is only one argument.
        iprint : bool
        """

        self.f1 = f1
        self.f2 = f2
        self.h = h
        self.args = args
        self.iprint = iprint

    def F(self, x, y):
        return np.array([
            self.f1(x, y, *self.args),
            self.f2(x, y, *self.args)
        ])

    def jacobian(self, x, y):
        h = self.h
        a = self.args

        df1_dx = (self.f1(x + h, y, *a) - self.f1(x - h, y, *a)) / (2 * h)
        df1_dy = (self.f1(x, y + h, *a) - self.f1(x, y - h, *a)) / (2 * h)

        df2_dx = (self.f2(x + h, y, *a) - self.f2(x - h, y, *a)) / (2 * h)
        df2_dy = (self.f2(x, y + h, *a) - self.f2(x, y - h, *a)) / (2 * h)

        return np.array([
            [df1_dx, df1_dy],
            [df2_dx, df2_dy]
        ])

    def hessian(self, f, x, y):
        h = self.h
        a = self.args

        dxx = (f(x + h, y, *a) - 2 * f(x, y, *a) + f(x - h, y, *a)) / h**2
        dyy = (f(x, y + h, *a) - 2 * f(x, y, *a) + f(x, y - h, *a)) / h**2
        dxy = (
            f(x + h, y + h, *a)
            - f(x + h, y - h, *a)
            - f(x - h, y + h, *a)
            + f(x - h, y - h, *a)
        ) / (4 * h**2)

        return np.array([
            [dxx, dxy],
            [dxy, dyy]
        ])

    def solve(self, x0, y0, tol=1e-10, max_iter=200):
        """
        Parameters
        ----------
        x0, y0 : float
            Initial guess
        tol : float
            Tolerance for convergence
        max_iter : int
            Maximum number of iterations

        Returns
        -------
        x, y : float
            Solution (x, y) if converged and stable, else (np.nan, np.nan)
        """

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
                    if self.iprint:
                        print("Unstable solution detected.")
                    return np.nan, np.nan ## Halley method converged but the solution is unstable
                # -------------------------

                return x, y ## Halley method converged with stable solution


            J = self.jacobian(x, y)

            try:
                delta = np.linalg.solve(J, F)
            except np.linalg.LinAlgError:
                if self.iprint:
                    print("Jacobian is singular.")
                return np.nan, np.nan

            H1 = self.hessian(self.f1, x, y)
            H2 = self.hessian(self.f2, x, y)

            H_delta = np.array([
                H1 @ delta,
                H2 @ delta
            ])

            correction = np.eye(2) - 0.5 * np.linalg.solve(J, H_delta)

            try:
                step = np.linalg.solve(correction, delta)
            except np.linalg.LinAlgError:
                step = delta  # fallback to Newton

            # damping
            alpha = 1.0
            while alpha > 1e-4:
                x_new = x - alpha * step[0]
                y_new = y - alpha * step[1]
                if np.linalg.norm(self.F(x_new, y_new)) < np.linalg.norm(F):
                    break
                alpha *= 0.5

            x, y = x_new, y_new

        if self.iprint:
            print("Halley method did not converge within the maximum number of iterations.")
        return np.nan, np.nan ## Halley method did not converge