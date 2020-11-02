-- script that lists all genres in the db htn_0d_tvshows_rate by rating
-- Each record should display: tv_genres.name - rating sum
-- Results must be sorted in descending order by their rating
-- You can use only one SELECT statement
-- The database name will be passed as an argument of the mysql command
SELECT tv_genres.name, SUM(tv_show_ratings.rate) AS rating FROM tv_genres
    JOIN tv_show_genres ON tv_genres.id = tv_show_genres.genre_id
    JOIN tv_show_ratings ON  tv_show_genres.show_id = tv_show_ratings.show_id
    GROUP BY tv_genres.name
    ORDER BY rating DESC;
