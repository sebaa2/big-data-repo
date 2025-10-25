-- 1. Distribución de clientes por región y frecuencia de compra
SELECT 
    region, 
    frecuencia_compra, 
    COUNT(*) AS cantidad_clientes
FROM storechile.clientes
GROUP BY region, frecuencia_compra
ORDER BY region, cantidad_clientes DESC;

-- 2. Ventas totales por región y método de pago
SELECT 
    region, 
    metodo_pago, 
    COUNT(*) AS total_ventas,
    SUM(precio) AS ingreso_total
FROM storechile.ventas
GROUP BY region, metodo_pago
ORDER BY ingreso_total DESC;

-- 3. Productos con menor stock (riesgo de desabastecimiento)
SELECT 
    nombre_producto, 
    categoria, 
    stock, 
    proveedor
FROM storechile.inventario
ORDER BY stock ASC
LIMIT 10;

-- 4. Tiempo promedio de sesión por página y acción
SELECT 
    pagina, 
    accion, 
    AVG(tiempo_sesion_segundos) AS tiempo_promedio_seg
FROM storechile.navegacion
GROUP BY pagina, accion
ORDER BY tiempo_promedio_seg DESC;

-- 5. Ventas por categoría de producto
SELECT 
    i.categoria, 
    COUNT(v.id_venta) AS total_ventas,
    SUM(v.precio) AS total_ingresos
FROM storechile.ventas v
JOIN storechile.inventario i ON v.producto LIKE CONCAT('%', i.nombre_producto, '%')
GROUP BY i.categoria
ORDER BY total_ingresos DESC;
