# -*- coding: utf-8 -*-
"""
Generic Weight Exporter for Fashion-MNIST SCNN
Uses a generic function to export trained PyTorch model weights to C header format.
"""

import os
import re
import torch
import torch.nn as nn
from typing import Iterable
import snntorch as snn
from snntorch import surrogate

def export_weights_to_c_header_generic(
    model: torch.nn.Module,
    weights_path: str,
    header_path: str = "network_weights.h",
    only_weights_and_bias: bool = True,   # True => exporta apenas *.weight/*.bias
    ctype: str = "weight_t",              # use "float" para já sair tipado em C
    emit_typedef_if_builtin: bool = True, # gera typedef weight_t se ctype in {"float","double"}
    line_wrap: int = 10,                  # quebra de linha a cada N valores
    float_fmt: str = ".8f",               # formato p/ floats (ex.: ".6g" p/ compacto)
    verbose: bool = True
):
    """
    Exporta pesos de QUALQUER arquitetura PyTorch para um .h compatível com C.

    Convenções de saída:
      - Tensores 1D (ex.: bias) -> vetor 1D
      - Tensores 2D (ex.: Linear.weight) -> matriz [dim0][dim1]
      - Tensores 4D (ex.: Conv2d.weight) -> vetor 1D flatten
      - Demais ranks (0D, 3D, 5D, ...) -> vetor 1D flatten
    Para cada tensor, emite #defines de shape: NAME_DIMi e NAME_NDIMS.
    """
    if verbose:
        print("Carregando pesos salvos...")

    ckpt = torch.load(weights_path, map_location="cpu")
    # aceita formatos variados de checkpoint
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        model.load_state_dict(ckpt["state_dict"])
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
    else:
        model.load_state_dict(ckpt)

    state = model.state_dict()

    # guard a partir do nome do arquivo
    base = os.path.basename(header_path)
    guard = re.sub(r"[^A-Za-z0-9]", "_", base).upper()

    def sanitize(k: str) -> str:
        # "layer1.0.conv1.weight" -> "layer1_0_conv1_weight"
        return re.sub(r"[^A-Za-z0-9_]", "_", k.replace(".", "_"))

    def should_keep(key: str) -> bool:
        if not only_weights_and_bias:
            return True
        return key.endswith(".weight") or key.endswith(".bias")

    def as_list_str(t: torch.Tensor) -> list[str]:
        # sempre grava como float no texto
        flat = t.detach().cpu().float().reshape(-1).tolist()
        fmt = "{:" + float_fmt + "}"
        return [fmt.format(v) for v in flat]

    def write_wrapped(f, values: Iterable[str], wrap=line_wrap, indent="  "):
        for i, v in enumerate(values):
            if i % wrap == 0:
                f.write("\n" + indent)
            f.write(v)
            if i != len(values) - 1:
                f.write(",")
        f.write("\n")

    if verbose:
        print(f"Exportando pesos para {header_path}...")

    stats = []

    with open(header_path, "w") as f:
        f.write(f"#ifndef {guard}\n")
        f.write(f"#define {guard}\n\n")

        # typedef opcional para weight_t
        if ctype in {"float", "double"} and emit_typedef_if_builtin:
            f.write(f"typedef {ctype} weight_t;\n\n")

        kept_items = [(k, v) for k, v in state.items() if torch.is_tensor(v) and should_keep(k)]
        f.write(f"// Número de tensores exportados\n")
        f.write(f"#define NN_NUM_TENSORS {len(kept_items)}\n\n")

        for key, tensor in kept_items:
            name = sanitize(key)
            t = tensor.detach().cpu().float()
            dims = list(t.shape)
            rank = t.dim()

            # Metadados do shape
            if dims:
                for i, d in enumerate(dims):
                    f.write(f"#define {name.upper()}_DIM{i} {d}\n")
            f.write(f"#define {name.upper()}_NDIMS {rank}\n\n")

            # Estatísticas
            mn = float(t.min().item()) if t.numel() else 0.0
            mx = float(t.max().item()) if t.numel() else 0.0
            stats.append((key, mn, mx))

            # Emissão por rank
            if rank == 2:
                rows, cols = dims
                f.write(f"{ctype} {name}[{rows}][{cols}] = {{\n")
                fmt = "{:" + float_fmt + "}"
                for i in range(rows):
                    row = t[i].reshape(-1).tolist()
                    f.write("  {")
                    for j, val in enumerate(row):
                        f.write(fmt.format(val))
                        if j != cols - 1:
                            f.write(",")
                    f.write("}")
                    if i != rows - 1:
                        f.write(",\n")
                    else:
                        f.write("\n")
                f.write("};\n\n")
            else:
                n = t.numel()
                vals = as_list_str(t)
                f.write(f"{ctype} {name}[{n}] = {{")
                write_wrapped(f, vals, wrap=line_wrap, indent="  ")
                f.write("};\n\n")

        f.write(f"#endif // {guard}\n")

    if verbose:
        print("Pesos exportados com sucesso!")
        print(f"Arquivo criado: {header_path}")
        for k, mn, mx in stats:
            print(f"Min {k}: {mn:{float_fmt}}")
            print(f"Max {k}: {mx:{float_fmt}}")


def create_scnn_complete(beta):
    """
    Recria a arquitetura da SCNN para carregamento dos pesos.
    Deve ser idêntica à função do arquivo original.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    spike_grad = surrogate.fast_sigmoid(slope=25)
    
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

        nn.MaxPool2d(2),
        nn.Dropout(0.25),

        nn.Conv2d(16, 32, 3, padding=1),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

        nn.MaxPool2d(2),
        nn.Dropout(0.25),

        nn.Conv2d(32, 64, 3, padding=1),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

        nn.MaxPool2d(2),
        nn.Dropout(0.4),

        nn.Flatten(),
        nn.Linear(64 * 3 * 3, 128),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

        nn.Dropout(0.4),
        nn.Linear(128, 10),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
    ).to(device)


def main():
    """
    Função principal para exportar os pesos do modelo Fashion-MNIST SCNN.
    """
    print("=" * 60)
    print("EXPORTADOR GENÉRICO DE PESOS - FASHION-MNIST SCNN")
    print("=" * 60)
    
    # Parâmetros do modelo (devem corresponder aos usados no treinamento)
    beta = 0.7
    weights_path = "trained_scnn_fashion_mnist_weights_only.pt"
    header_path = "fashion_mnist_weights_generic.h"
    
    # Verificar se o arquivo de pesos existe
    if not os.path.exists(weights_path):
        print(f"Erro: Arquivo de pesos '{weights_path}' não encontrado!")
        print("Execute o treinamento primeiro ou verifique o caminho do arquivo.")
        return
    
    print(f"Arquivo de pesos encontrado: {weights_path}")
    
    # Criar o modelo (deve ter a mesma arquitetura usada no treinamento)
    print("Criando modelo SCNN...")
    model = create_scnn_complete(beta)
    
    # Exportar os pesos usando a função genérica
    try:
        export_weights_to_c_header_generic(
            model=model,
            weights_path=weights_path,
            header_path=header_path,
            only_weights_and_bias=True,      # Exporta apenas weights e bias
            ctype="float",                   # Tipo C (float direto)
            emit_typedef_if_builtin=False,   # Não emite typedef para float
            line_wrap=10,                    # 10 valores por linha
            float_fmt=".8f",                 # Formato de precisão
            verbose=True                     # Saída detalhada
        )
        
        print(f"\n✓ Exportação concluída com sucesso!")
        print(f"✓ Arquivo criado: {header_path}")
        print(f"✓ O arquivo está pronto para uso em implementações C/C++")
        
    except Exception as e:
        print(f"Erro durante a exportação: {e}")
        return
    
    print("=" * 60)


if __name__ == "__main__":
    main()
