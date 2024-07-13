from bs4 import BeautifulSoup
import requests
import pandas as pd

euro_years = [1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020]


def find_matches(year):
    if year !=2024:
        web=f'https://en.wikipedia.org/wiki/UEFA_Euro_{year}'
    else:
        web='https://web.archive.org/web/20240602150318/https://en.wikipedia.org/wiki/UEFA_Euro_2024'

    response=requests.get(web)
    content = response.text

    soup= BeautifulSoup(content, 'lxml')

    euro_matches=soup.find_all('div',class_='footballbox')

    home_team=[]
    game_score=[]
    away_team=[]

    for match in euro_matches:
        home_team.append(match.find('th', class_='fhome').get_text())
        game_score.append(match.find('th', class_='fscore').get_text())
        away_team.append(match.find('th', class_='faway').get_text())

    football_dict = {'Home Team': home_team, 'Game Score': game_score, 'Away Team': away_team}

    euro_df=pd.DataFrame(football_dict)

    euro_df['Year'] = year
    return euro_df

euro_results = [find_matches(year) for year in euro_years]

euro_df=pd.concat(euro_results, ignore_index=True)
euro_df.to_csv('Euro_results.csv', index=False)

df_fixture=find_matches(2024)
df_fixture.to_csv('euro_fixture.csv', index=False)