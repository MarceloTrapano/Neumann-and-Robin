from Robin import Robin
import matplotlib.pyplot as plt
def main() -> None:
    alpha =     lambda x: 3*x**2
    beta =      lambda x: (1+x)**2
    gamma =     lambda x: -x**2
    f =         lambda x: 1+2*x**2

    a = [2, -3, 1, 0]
    b = [6, 0, -1, 2]

    robin = Robin(alpha, beta, gamma, f)

    robin.setAB(a, b)
    robin.plt_config()
    robin.setLabel(r"$3x^2\frac{d^2y}{dx^2}+(1+x)^2\frac{dy}{dx}-x^2y= 1+2x^2$")
    for n in [10_000, 50_000, 100_000, 500_000, 1_000_000]:
        x = robin.setN(n)
        sol = robin.solve()
        label = f"Aproksymacja dla n = {n}"
        plt.plot(x, sol, label=label)
    plt.grid(True)
    plt.ylabel(r"Wartości funkcji $f(x)$")
    plt.xlabel(r"Wartości $x$")
    plt.title(robin.label)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()