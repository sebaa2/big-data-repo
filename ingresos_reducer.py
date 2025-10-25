#!/usr/bin/env python3
import sys

def reducer():
    current_producto = None
    total_ingresos = 0
    
    for line in sys.stdin:
        # Dividir la línea en clave y valor
        producto, precio_str = line.strip().split('\t')
        precio = float(precio_str)
        
        if current_producto == producto:
            # Mismo producto, acumular ingresos
            total_ingresos += precio
        else:
            # Nuevo producto, emitir el anterior
            if current_producto:
                print(f"{current_producto}\t{total_ingresos:.2f}")
            
            # Reiniciar para el nuevo producto
            current_producto = producto
            total_ingresos = precio
    
    # No olvidar el último producto
    if current_producto:
        print(f"{current_producto}\t{total_ingresos:.2f}")

if __name__ == "__main__":
    reducer()
