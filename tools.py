from langchain.agents import Tool

def create_tools(chain, agent_executor_sql):

    tools_mix = [
        Tool(
            name='RetrieverAgent',
            func=chain.invoke,
            description='''
            Useful to search movie by description and description by movie.
            Also useful for finding similar movies.
            Also useful to find actors of the movie by movie title ONLY IF HAVING LESS THEN 5 MOVIES! (but if SQLAgent did't work you can still use it if query has more than 5 movies)
            Also useful to find movies by genre if SQLAgent didn't help.
            '''
        ),
        Tool(
            name="SQLAgent",
            func=agent_executor_sql.invoke,
            description="""
            Useful to query sql tables, search and sort movies by genre, actors, directors, movie rating.
            Useful when you are searching for movies that are the best or worst.
            Useful when it is expected a several movies output (5 and more).
            Genres are all in russian with lowercase letters, table actors has column actor_name with actor names in English like "Ryan Reynolds".
            Also useful to search movie if RetrieverAgent didn't help.
            In case title of movies can be the same for different movies it's better to check them by movie_id if possible.
            By default also output rating for every movie if possible.
            If you are searching for more than 5 movies don't use 'SELECT *...', instead SELECT only columns that needed.
            There is an example of sql query to this database:
            'SELECT title_ru, genre_name, actor_name FROM movies JOIN movie_genres using(movie_id) JOIN genres using(genre_id) JOIN movie_actors using(movie_id) JOIN actors using(actor_id)'
            but it is just an example that joins tables from this database and selects only several columns, you should select columns that you need.
            the action input have to be like in example format with No 'sql' word or quotes
            Action Input: SELECT title_ru, rating FROM movies
            JOIN movie_genres using(movie_id)
            JOIN genres using(genre_id)
            WHERE genre_name = 'триллер'
            ORDER BY rating DESC
            LIMIT 5;
            """,
        )
    ]

    return tools_mix

