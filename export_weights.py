import os
import re
import torch
import torch.nn as nn
from typing import Iterable
import snntorch as snn
from snntorch import surrogate

def export_weights_to_c_header_generic(
    model: torch.nn.Module | None,
    weights_path: str,
    header_path: str = "network_weights.h",
    only_weights_and_bias: bool = True,   # True => exporta apenas *.weight/*.bias
    ctype: str = "weight_t",              # use "float" para já sair tipado em C
    emit_typedef_if_builtin: bool = True, # gera typedef weight_t se ctype in {"float","double"}
    line_wrap: int = 10,                  # quebra de linha a cada N valores
    float_fmt: str = ".8f",               # formato p/ floats (ex.: ".6g" p/ compacto)
    verbose: bool = True,
    require_model: bool = True            # <<< NOVO: se False, usa apenas o state_dict do arquivo
):
    """
    Exporta pesos de QUALQUER arquitetura PyTorch para um .h compatível com C.

    Convenções de saída:
      - Tensores 1D (ex.: bias) -> vetor 1D
      - Tensores 2D (ex.: Linear.weight) -> matriz [dim0][dim1]
      - Tensores 4D (ex.: Conv2d.weight) -> vetor 1D flatten
      - Demais ranks (0D, 3D, 5D, ...) -> vetor 1D flatten
    Para cada tensor, emite #defines de shape: NAME_DIMi e NAME_NDIMS.

    Quando require_model=False, o `model` pode ser None; o exportador usa o state_dict
    salvo em `weights_path` diretamente (aceita formatos comuns de checkpoint).
    """
    if verbose:
        print("Carregando pesos salvos...")

    # Carrega o checkpoint cru
    ckpt = torch.load(weights_path, map_location="cpu")

    # Função auxiliar para extrair um state_dict de diferentes formatos
    def extract_state_dict(obj):
        # Formato comum de trainers: {"state_dict": {...}, ...}
        if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        # Checkpoint já é um state_dict puro (chaves com .weight/.bias)
        if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
            return obj
        # Último recurso: tenta acessar .state_dict() se for um módulo salvo
        if hasattr(obj, "state_dict") and callable(getattr(obj, "state_dict")):
            return obj.state_dict()
        raise ValueError("Formato de checkpoint não reconhecido para extração de state_dict.")

    if require_model:
        if model is None:
            raise ValueError("`model` é obrigatório quando require_model=True.")
        # carrega no modelo (mantém comportamento anterior)
        if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            model.load_state_dict(ckpt["state_dict"])
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
        else:
            model.load_state_dict(ckpt)
        state = model.state_dict()
    else:
        # usa apenas o state_dict do arquivo (sem precisar do model)
        state = extract_state_dict(ckpt)

    # guard a partir do nome do arquivo
    base = os.path.basename(header_path)
    guard = re.sub(r"[^A-Za-z0-9]", "_", base).upper()

    def sanitize(k: str) -> str:
        # "layer1.0.conv1.weight" -> "layer1_0_conv1_weight"
        sanitized = re.sub(r"[^A-Za-z0-9_]", "_", k.replace(".", "_"))
        # Se começar com dígito, adiciona prefixo "layer_"
        if sanitized and sanitized[0].isdigit():
            sanitized = "layer_" + sanitized
        return sanitized

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
        values = list(values)  # garantir len()
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
            elif rank == 4:
                # Caso especial para Conv2d: [out_channels, in_channels, kernel_h, kernel_w]
                out_ch, in_ch, kh, kw = dims
                f.write(f"{ctype} {name}[{out_ch}][{in_ch}][{kh}][{kw}] = {{\n")
                fmt = "{:" + float_fmt + "}"
                
                for out_idx in range(out_ch):
                    f.write("  {\n")
                    for in_idx in range(in_ch):
                        f.write("    {\n")
                        for h_idx in range(kh):
                            f.write("      {")
                            for w_idx in range(kw):
                                val = t[out_idx, in_idx, h_idx, w_idx].item()
                                f.write(fmt.format(val))
                                if w_idx != kw - 1:
                                    f.write(",")
                            f.write("}")
                            if h_idx != kh - 1:
                                f.write(",\n")
                            else:
                                f.write("\n")
                        f.write("    }")
                        if in_idx != in_ch - 1:
                            f.write(",\n")
                        else:
                            f.write("\n")
                    f.write("  }")
                    if out_idx != out_ch - 1:
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
    Para extrair os pesos de um arquivo ".pt" sem o modelo, defina require_model = False?
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
    # Parâmetros do modelo (devem corresponder aos usados no treinamento)
    beta = 0.7
    weights_path = "weight_input.pt"
    
    # Verificar se o arquivo de pesos existe
    if not os.path.exists(weights_path):
        print(f"Erro: Arquivo de pesos '{weights_path}' não encontrado!")
        print("Execute o treinamento primeiro ou verifique o caminho do arquivo.")
        return
    
    print(f"Arquivo de pesos encontrado: {weights_path}")
    
    # MODO 1: Com modelo (require_model=True) - Comportamento original
    print("\n" + "=" * 40)
    print("MODO 1: EXPORTAÇÃO COM MODELO")
    print("=" * 40)
    
    try:
        print("Criando modelo SCNN...")
        model = create_scnn_complete(beta)
        
        export_weights_to_c_header_generic(
            model=model,
            weights_path=weights_path,
            header_path="fashion_mnist_weights_with_model.h",
            only_weights_and_bias=True,      # Exporta apenas weights e bias
            ctype="float",                   # Tipo C (float direto)
            emit_typedef_if_builtin=False,   # Não emite typedef para float
            line_wrap=10,                    # 10 valores por linha
            float_fmt=".8f",                 # Formato de precisão
            verbose=True,                    # Saída detalhada
            require_model=True               # Usa o modelo
        )
        
        print(f"✓ Modo 1 concluído: fashion_mnist_weights_with_model.h")
        
    except Exception as e:
        print(f"Erro no Modo 1: {e}")
    
    # MODO 2: Sem modelo (require_model=False) - Novo comportamento
    print("\n" + "=" * 40)
    print("MODO 2: EXPORTAÇÃO SEM MODELO")
    print("=" * 40)
    
    try:
        export_weights_to_c_header_generic(
            model=None,                      # Sem modelo
            weights_path=weights_path,
            header_path="weights_output.h",
            only_weights_and_bias=True,      # Exporta apenas weights e bias
            ctype="float",                   # Tipo C (float direto)
            emit_typedef_if_builtin=False,   # Não emite typedef para float
            line_wrap=10,                    # 10 valores por linha
            float_fmt=".8f",                 # Formato de precisão
            verbose=True,                    # Saída detalhada
            require_model=False              # NÃO usa o modelo
        )
        export_weights_to_c_header_generic(
        model=None,                          # Sem modelo
        weights_path=weights_path,           # Apenas o arquivo de pesos
        header_path="simple_export.h",      # Arquivo de saída
        require_model=False                  # Modo direto
    )
        
        print(f"✓ Modo 2 concluído: fashion_mnist_weights_direct.h")
        
    except Exception as e:
        print(f"Erro no Modo 2: {e}")
    

if __name__ == "__main__":
    # Executa a demonstração completa
    main()
