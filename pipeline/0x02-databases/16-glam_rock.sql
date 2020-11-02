-- script that lists all bands with Glam as their main style, ranked by their longevity
SELECT band_name , IF(split is NULL, YEAR(CURDATE()), split) - formed AS lifespan FROM metal_bands
	WHERE style like '%Glam Rock%'
	ORDER BY lifespan DESC, band_name DESC;
