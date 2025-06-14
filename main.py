import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

class ODESolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Численное решение ОДУ")
        self.root.geometry("1200x800")

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        self.params_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.params_frame, text='Параметры')

        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text='Результаты')

        self.plot_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plot_frame, text='Графики')

        self.setup_params_tab()
        self.setup_results_tab()
        self.setup_plot_tab()

        self.load_demo_task()

    def setup_params_tab(self):
        frame = ttk.LabelFrame(self.params_frame, text="Уравнение")
        frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(frame, text="dy/dt = f(t, y) =").grid(row=0, column=0)
        self.eq_entry = ttk.Entry(frame, width=50)
        self.eq_entry.grid(row=0, column=1, padx=5, pady=5)

        init_frame = ttk.LabelFrame(self.params_frame, text="Начальные условия")
        init_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(init_frame, text="t₀:").grid(row=0, column=0)
        self.t0_entry = ttk.Entry(init_frame, width=10)
        self.t0_entry.grid(row=0, column=1, padx=5)

        ttk.Label(init_frame, text="y₀:").grid(row=0, column=2)
        self.y0_entry = ttk.Entry(init_frame, width=10)
        self.y0_entry.grid(row=0, column=3, padx=5)

        solve_frame = ttk.LabelFrame(self.params_frame, text="Параметры решения")
        solve_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(solve_frame, text="Конечное время (t_end):").grid(row=0, column=0)
        self.t_end_entry = ttk.Entry(solve_frame, width=10)
        self.t_end_entry.grid(row=0, column=1, padx=5)

        ttk.Label(solve_frame, text="Шаг (h):").grid(row=0, column=2)
        self.h_entry = ttk.Entry(solve_frame, width=10)
        self.h_entry.grid(row=0, column=3, padx=5)

        params_frame = ttk.LabelFrame(self.params_frame, text="Параметры задачи (через запятую)")
        params_frame.pack(fill='x', padx=10, pady=5)

        self.params_entry = ttk.Entry(params_frame, width=50)
        self.params_entry.pack(padx=5, pady=5, fill='x')

        button_frame = ttk.Frame(self.params_frame)
        button_frame.pack(fill='x', padx=10, pady=10)

        ttk.Button(button_frame, text="Демо-задача", command=self.load_demo_task).pack(side='left')
        ttk.Button(button_frame, text="Решить", command=self.solve_equation).pack(side='left')
        ttk.Button(button_frame, text="Очистить", command=self.clear_all).pack(side='right')

    def setup_results_tab(self):
        self.results_text = scrolledtext.ScrolledText(self.results_frame, wrap=tk.WORD)
        self.results_text.pack(fill='both', expand=True, padx=10, pady=10)
        self.results_text.config(state=tk.DISABLED)

    def setup_plot_tab(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.tight_layout(pad=5.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

    def load_demo_task(self):
        self.clear_all()
        self.eq_entry.insert(0, "-p0 * y")
        self.t0_entry.insert(0, "0")
        self.y0_entry.insert(0, "1000")
        self.t_end_entry.insert(0, "30")
        self.h_entry.insert(0, "0.5")
        self.params_entry.insert(0, "0.1")

        self.update_results("Демо-задача: Радиоактивный распад\n")
        self.update_results("Уравнение: dN/dt = -λN\n")
        self.update_results("Параметры: λ = 0.1 мин⁻¹, N₀ = 1000 г\n")
        self.update_results("Аналитическое решение: N(t) = N₀ * exp(-λt)\n")

    def clear_all(self):
        self.eq_entry.delete(0, tk.END)
        self.t0_entry.delete(0, tk.END)
        self.y0_entry.delete(0, tk.END)
        self.t_end_entry.delete(0, tk.END)
        self.h_entry.delete(0, tk.END)
        self.params_entry.delete(0, tk.END)
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)
        self.ax1.clear()
        self.ax2.clear()
        self.canvas.draw()

    def update_results(self, text):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.insert(tk.END, text)
        self.results_text.config(state=tk.DISABLED)
        self.results_text.see(tk.END)

    def euler_method(self, f, t_span, y0, h):
        t0, tf = t_span
        n_steps = int((tf - t0) / h) + 1
        t = np.linspace(t0, tf, n_steps)
        y = np.zeros(n_steps)
        y[0] = y0

        for i in range(1, n_steps):
            y[i] = y[i-1] + h * f(t[i-1], y[i-1])
        return t, y

    def rk4_method(self, f, t_span, y0, h):
        t0, tf = t_span
        n_steps = int((tf - t0) / h) + 1
        t = np.linspace(t0, tf, n_steps)
        y = np.zeros(n_steps)
        y[0] = y0

        for i in range(1, n_steps):
            k1 = f(t[i-1], y[i-1])
            k2 = f(t[i-1] + h/2, y[i-1] + h/2 * k1)
            k3 = f(t[i-1] + h/2, y[i-1] + h/2 * k2)
            k4 = f(t[i-1] + h, y[i-1] + h * k3)
            y[i] = y[i-1] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        return t, y

    def solve_equation(self):
        try:
            eq_str = self.eq_entry.get()
            t0 = float(self.t0_entry.get())
            y0 = float(self.y0_entry.get())
            t_end = float(self.t_end_entry.get())
            h = float(self.h_entry.get())
            params_str = self.params_entry.get()

            params = []
            if params_str:
                params = [float(p.strip()) for p in params_str.split(',')]

            def f(t, y):
                local_vars = {'t': t, 'y': y}
                for i, p in enumerate(params):
                    local_vars[f'p{i}'] = p
                local_vars.update({'sin': np.sin, 'cos': np.cos, 'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt, 'pi': np.pi})
                return eval(eq_str, {"__builtins__": None}, local_vars)

            test_val = f(t0, y0)
            if not isinstance(test_val, (int, float)):
                raise ValueError("Функция должна возвращать числовое значение")

            t_span = (t0, t_end)
            t_euler, y_euler = self.euler_method(f, t_span, y0, h)
            t_rk4, y_rk4 = self.rk4_method(f, t_span, y0, h)

            self.display_results(t_euler, y_euler, t_rk4, y_rk4)
            self.plot_results(t_euler, y_euler, t_rk4, y_rk4)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при решении уравнения: {str(e)}")

    def display_results(self, t_euler, y_euler, t_rk4, y_rk4):
        self.update_results("\n=== РЕЗУЛЬТАТЫ ===\n")
        self.update_results(f"Число точек: {len(t_euler)}\n")
        self.update_results(f"Финальное значение (Эйлер): {y_euler[-1]:.6f}\n")
        self.update_results(f"Финальное значение (Рунге-Кутта): {y_rk4[-1]:.6f}\n")
        self.update_results("\nПервые 5 точек:\n")
        self.update_results(" t       | Эйлер     | Рунге-Кутта\n")
        self.update_results("---------|-----------|------------\n")
        for i in range(min(5, len(t_euler))):
            self.update_results(f"{t_euler[i]:<8.3f}| {y_euler[i]:<10.6f}| {y_rk4[i]:<10.6f}\n")

    def plot_results(self, t_euler, y_euler, t_rk4, y_rk4):
        self.ax1.clear()
        self.ax2.clear()

        self.ax1.plot(t_euler, y_euler, 'b-', label='Метод Эйлера')
        self.ax1.plot(t_rk4, y_rk4, 'r--', label='Метод Рунге-Кутта 4')
        self.ax1.set_xlabel('Время (t)')
        self.ax1.set_ylabel('y(t)')
        self.ax1.set_title('Сравнение методов')
        self.ax1.legend()
        self.ax1.grid(True)

        error = np.abs(y_euler - y_rk4[:len(y_euler)])
        self.ax2.plot(t_euler, error, 'g-', label='Разница')
        self.ax2.set_xlabel('Время (t)')
        self.ax2.set_ylabel('|Эйлер - Рунге-Кутта|')
        self.ax2.set_title('Ошибка между методами')
        self.ax2.legend()
        self.ax2.grid(True)
        if np.max(error) > 0:
            self.ax2.set_yscale('log')

        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = ODESolverApp(root)
    root.mainloop()