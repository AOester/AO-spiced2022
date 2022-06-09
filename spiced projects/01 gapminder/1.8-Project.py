import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import imageio

fert = pd.read_csv('data/gapminder_total_fertility.csv', index_col=0)

life = pd.read_excel('data/gapminder_lifeexpectancy.xlsx', index_col=0)

pop = pd.read_excel('data/gapminder_population.xlsx', index_col=0)
#print("Fruchtbarkeit",fert.shape)
#print("Lebenserwartung",life.shape)

#print(fert.columns)
#print(life.columns)

fert.columns = fert.columns.astype(int)
#wandelt die String Bezeichnung der Spalten (Jahreszahlen hier) in integer


#print(fert.columns)
#print(life.columns)

#print(fert.index)

fert.index.name = 'country'
#gibt dem Index den Namen 'country'
fert = fert.reset_index()
#erschafft einen neuen Index, der alte Index wird zur ersten Spalte (Spalte 0)

fert_lon = fert.melt(id_vars='country', var_name='year', value_name='fertility_rate')
# .melt wandelt wide format zu long format

life.index.name = 'country'
#gibt dem Index den Namen 'country'
life = life.reset_index()
#erschafft einen neuen Index, der alte Index wird zur ersten Spalte (Spalte 0)

life_lon = life.melt(id_vars='country', var_name='year', value_name='life_expectancy')
# .melt wandelt wide format zu long format

pop.index.name = 'country'
pop = pop.reset_index()
pop_lon = pop.melt(id_vars='country', var_name='year', value_name='Total population')

# (1) fert Fertility wide format
# (2) life Life expectancy wide format
# (3) pop Population wide format

# (1) fert_lon Fertility long format
# (2) life_lon Life expectancy long format
# (3) pop_lon Population long format

# merge (1) + (2)
fert_life = fert_lon.merge(life_lon)
# merge + (3)
fert_life_pop = fert_life.merge(pop_lon)

flpnona = fert_life_pop
#flpnona = fert_life_pop.dropna()
flpnona.reset_index(inplace=True)
flpnona=flpnona.drop('index',axis=1)
conti = pd.read_csv('data/continents.csv', delimiter = ';', index_col=0)
conti = conti.reset_index()
fulltab = flpnona.merge(conti, on = "country" , how='left' )
contilist = conti.continent.unique()

plt.style.use('seaborn-dark')
sns.set_style('darkgrid')
sns.color_palette('tab10')

images=[]

for i in range(1960,2016) :
    plt.title(i)
    fert_life_pop_subset = fulltab.loc[fulltab['year'] == i]
    ax = sns.scatterplot(x='life_expectancy', y='fertility_rate' ,palette='tab10', hue = 'continent',size='Total population' , data=fert_life_pop_subset, alpha=0.75)
    plt.axis((10,90,0,9))
    
    # plt.legend( bbox_to_anchor=(0, 0), loc=3, )

    handles, legend_handle_list = ax.get_legend_handles_labels()

    ax.legend(handles=handles[1:],labels=legend_handle_list[1:7],bbox_to_anchor=(0, 0), loc=3)

    filename = 'images/life-fert-VS-{0}.png'.format(i)

    plt.savefig(filename,dpi=200)
    
    images.append(imageio.imread(filename))
    plt.close()

imageio.mimsave('images/Gapminder.gif', images, fps=10)