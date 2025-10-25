#!/usr/bin/env python3
import sys

def mapper():
    for line in sys.stdin:
        # Saltar la línea de encabezado
        if "id_venta,producto,precio,metodo_pago,fecha,region" in line:
            continue
            
        # Dividir la línea en campos
        fields = line.strip().split(',')
        
        # Verificar que tenemos al menos 6 campos
        if len(fields) >= 6:
            try:
                producto = fields[1].strip()
                precio = float(fields[2].strip())
                
                # Emitir: producto -> precio
                print(f"{producto}\t{precio}")
                
            except ValueError:
                # Si hay error en conversión, saltar línea
                continue

if __name__ == "__main__":
    mapper()
