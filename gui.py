import tkinter as tk


class Grid:
    def __init__(self, master):
        self.master = master
        self.grid = [[0 for _ in range(5)] for _ in range(5)]
        self.buttons = [
            [tk.Button(master, bg='white', command=lambda i=i, j=j: self.change_color(i, j), width=5, height=2) for j in
             range(5)] for i in
            range(5)]
        for i in range(5):
            for j in range(5):
                self.buttons[i][j].grid(row=i, column=j)

    def change_color(self, i, j):
        if self.grid[i][j] == 0:
            self.grid[i][j] = 1
            self.buttons[i][j].config(bg='black')
        else:
            self.grid[i][j] = 0
            self.buttons[i][j].config(bg='white')

    def reset(self):
        self.grid = [[0 for _ in range(5)] for _ in range(5)]
        for i in range(5):
            for j in range(5):
                self.buttons[i][j].config(bg='white')


class App:
    def __init__(self, master):
        self.master = master
        self.grid = Grid(master)
        self.button = tk.Button(master, text='Print Array', command=self.print_array)
        self.button.grid(row=5, column=0, columnspan=5)
        self.reset_button = tk.Button(master, text='Reset', command=self.reset)
        self.reset_button.grid(row=6, column=0, columnspan=5)

    def print_array(self):
        # l = list()
        # for i in self.grid.grid:
        #     for j in i:
        #         l.append(j)
        # print(l)
        print(self.grid.grid)

    def reset(self):
        self.grid.reset()


if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
