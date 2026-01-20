import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig, norm
import time

#-----------QR metoda za simetrične matrike----------
def qr_metoda(A, tol=1e-1, max_iter=500000):
    A_k = A.copy()
    history = []
    for i in range(max_iter):
        Q, R = np.linalg.qr(A_k)
        A_k = R @ Q
        history.append(np.sort(np.diagonal(A_k)))
        if np.allclose(A_k - np.diag(np.diagonal(A_k)), 0, atol=tol):
            break
    return np.diagonal(A_k), i + 1, np.array(history)

# ---------Householderjeva tridiagonalizacija-------
def householder_trig(A):
    A = A.copy()
    n = A.shape[0]
    for k in range(n - 2):
        x = A[k+1:, k]
        e = np.zeros_like(x)
        e[0] = norm(x)
        u = x - e if x[0] >= 0 else x + e
        u /= norm(u)
        H = np.eye(n)
        H_k = np.eye(n - k - 1) - 2.0 * np.outer(u, u)
        H[k+1:, k+1:] = H_k
        A = H @ A @ H.T
    return A

# ---------Grafi-------
def plot_convergence(history, title, builtin_vals=None):
    for i in range(history.shape[1]):
        plt.plot(history[:, i], label=f'λ{i+1} (QR)')
        if builtin_vals is not None:
            plt.hlines(builtin_vals[i], 0, len(history), colors='k', linestyles='dotted', alpha=0.6, label=f'λ{i+1} (vgrajena)')
    plt.title(title)
    plt.xlabel("Iteracija")
    plt.ylabel("Lastne vrednosti")
    plt.grid(True)
    plt.legend()
    plt.show()
def plot_all_convergences(hist_qr, hist_tri, builtin_vals, title):
    iterations_qr = hist_qr.shape[0]
    iterations_tri = hist_tri.shape[0]

    plt.figure(figsize=(10, 6))
    for i in range(hist_qr.shape[1]):
        plt.plot(hist_qr[:, i], label=f'QR λ{i+1}', linestyle='-')
        plt.plot(range(iterations_tri), hist_tri[:, i], label=f'trig_QR λ{i+1}', linestyle='--')
        plt.hlines(builtin_vals[i], 0, max(iterations_qr, iterations_tri), colors='k', linestyles='dotted', label=f'Vgrajena λ{i+1}')

    plt.title(title)
    plt.xlabel("Iteracija")
    plt.ylabel("Lastne vrednosti")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



#-----------Čas in lastne vrednosti----------
def test_qr_on_matrix(A, name="Matrika"):
    print(f"\n--- {name} ---")

    # Vgrajena metoda
    start_builtin = time.time()
    builtin_vals = np.sort(eig(A)[0])
    end_builtin = time.time()
    print("Vgrajene lastne vrednosti:", builtin_vals)
    print(f"Čas vgrajene metode: {(end_builtin - start_builtin)*1000:.3f} ms")

    # QR metoda brez tridiagonalizacije
    start_qr = time.time()
    vals_qr, iters_qr, hist_qr = qr_metoda(A)
    end_qr = time.time()
    print(f"QR brez tridiagonalizacije ({iters_qr} iteracij):", np.sort(vals_qr))
    print(f"Čas QR brez tridiagonalizacije: {(end_qr - start_qr)*1000:.3f} ms")

    # QR metoda s tridiagonalizacijo
    A_tri = householder_trig(A)
    start_qr_tri = time.time()
    vals_tri, iters_tri, hist_tri = qr_metoda(A_tri)
    end_qr_tri = time.time()
    print(f"QR s tridiagonalizacijo ({iters_tri} iteracij):", np.sort(vals_tri))
    print(f"Čas QR s tridiagonalizacijo: {(end_qr_tri - start_qr_tri)*1000:.3f} ms")

    # Vizualizacije
    plot_convergence(hist_qr, f"{name} - QR brez tridiagonalizacije", builtin_vals)
    plot_convergence(hist_tri, f"{name} - QR s tridiagonalizacijo", builtin_vals)
    plot_all_convergences(hist_qr, hist_tri, builtin_vals, f"{name} – primerjava vseh metod")


# -------Primer matrike-----------
# A = np.array([
#     [1000., 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
#         [0.1, 1000., 0.1, 0.01, 0.001, 0.0001, 0.00001],
#         [0.01, 0.1, 1000., 0.1, 0.01, 0.001, 0.0001],
#         [0.001, 0.01, 0.1, 1000., 0.1, 0.01, 0.001],
#         [0.0001, 0.001, 0.01, 0.1, 1000., 0.1, 0.01],
#         [0.00001, 0.0001, 0.001, 0.01, 0.1, 1000., 0.1],
#         [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1000.]
# ]) 
A = np.array([
        [1e6, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
        [0.1, 1e6, 0.1, 0.01, 0.001, 0.0001, 0.00001],
        [0.01, 0.1, 1e6, 0.1, 0.01, 0.001, 0.0001],
        [0.001, 0.01, 0.1, 1e6, 0.1, 0.01, 0.001],
        [0.0001, 0.001, 0.01, 0.1, 1e6, 0.1, 0.01],
        [0.00001, 0.0001, 0.001, 0.01, 0.1, 1e6, 0.1],
        [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1e6]
    ])

test_qr_on_matrix(A, "Simetrična matrika A")

