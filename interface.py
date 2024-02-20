import tkinter as tk
from tkinter import ttk
from abordagem_A import  Render
from abordagem_B import  Render_Class_B
from abordagem_C import  Render_Class_C
   
# Janela principal
janela_principal = tk.Tk()
janela_principal.title("Abordagens para o ambiente desconhecido")

# Label de boas-vindas
tk.Label(janela_principal, text="BEM-VINDO A SIMULAÇÃO DE AGENTES", font=("Arial bold", 20)).pack(pady=5, padx=5)
tk.Label(janela_principal, text="Escolha uma abordagem para iniciar:").pack()

# Botões para abrir as abordagens
btn_abordagem1 = tk.Button(janela_principal, text="Abordagem A", command=Render.renderizar_Abordagem_A)
btn_abordagem1.pack(pady=5)
btn_abordagem2 = tk.Button(janela_principal, text="Abordagem B", command=Render_Class_B.renderizar_Abordagem_B)
btn_abordagem2.pack(pady=5)
btn_abordagem3 = tk.Button(janela_principal, text="Abordagem C", command=Render_Class_C.renderizar_Abordagem_C)
btn_abordagem3.pack(pady=5)

janela_principal.mainloop()
    