-- function that safely divides
delimiter $$
CREATE FUNCTION SafeDiv(a INT, b INT)
	RETURNS FLOAT
	BEGIN
		RETURN (IF (b = 0, 0, a / b));
	END $$
delimiter ;
