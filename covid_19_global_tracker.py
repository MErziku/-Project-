"""
COVID-19 Global Data Tracker
Analysis of global COVID-19 data including cases, deaths, and vaccinations
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('ggplot')
sns.set_palette("husl")

def load_data():
    """Load the COVID-19 dataset from Our World in Data"""
    try:
        # Try to load from local file
        df = pd.read_csv('owid-covid-data.csv')
        print("Dataset loaded from local file.")
    except FileNotFoundError:
        try:
            # If local file not found, try to download from Our World in Data
            url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
            df = pd.read_csv(url)
            print("Dataset downloaded from Our World in Data.")
            # Save a local copy for future use
            df.to_csv('owid-covid-data.csv', index=False)
            print("Dataset saved locally for future use.")
        except Exception as e:
            print(f"Could not load the dataset: {e}")
            return None
            
    return df

def clean_data(df):
    """Clean and preprocess the COVID-19 data"""
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Filter for countries only (remove regional aggregates)
    aggregates = ['World', 'Asia', 'Europe', 'North America', 'South America', 
                  'European Union', 'Africa', 'Oceania', 'International']
    df = df[~df['location'].isin(aggregates)]

    # Select key columns for analysis
    key_columns = ['date', 'location', 'continent', 'population', 
                   'total_cases', 'new_cases', 'total_deaths', 'new_deaths',
                   'total_cases_per_million', 'new_cases_per_million',
                   'total_deaths_per_million', 'new_deaths_per_million',
                   'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated',
                   'total_boosters', 'new_vaccinations', 'population_density',
                   'median_age', 'aged_65_older', 'aged_70_older',
                   'gdp_per_capita', 'extreme_poverty', 'cardiovasc_death_rate',
                   'diabetes_prevalence', 'female_smokers', 'male_smokers',
                   'handwashing_facilities', 'hospital_beds_per_thousand',
                   'life_expectancy', 'human_development_index']

    # Create a working dataframe with key columns
    covid_df = df[key_columns].copy()

    # Fill missing values for specific columns with 0
    columns_to_fill_zero = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths',
                           'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated',
                           'total_boosters', 'new_vaccinations']

    for col in columns_to_fill_zero:
        covid_df[col] = covid_df[col].fillna(0)

    # For other numeric columns, fill with median of the continent
    numeric_cols = covid_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in columns_to_fill_zero + ['population']]

    for col in numeric_cols:
        covid_df[col] = covid_df.groupby('continent')[col].transform(lambda x: x.fillna(x.median()))

    # Calculate additional metrics
    covid_df['death_rate'] = covid_df['total_deaths'] / covid_df['total_cases']
    covid_df['death_rate'] = covid_df['death_rate'].replace([np.inf, -np.inf], np.nan)

    covid_df['cases_per_million'] = covid_df['total_cases'] / (covid_df['population'] / 1e6)
    covid_df['deaths_per_million'] = covid_df['total_deaths'] / (covid_df['population'] / 1e6)

    # Filter for countries with sufficient data
    min_cases = 1000
    countries_with_sufficient_data = covid_df[covid_df['total_cases'] > min_cases]['location'].unique()
    covid_df = covid_df[covid_df['location'].isin(countries_with_sufficient_data)]

    print(f"Data cleaned. Working with {covid_df.shape[0]} rows and {covid_df.shape[1]} columns.")
    return covid_df

def exploratory_data_analysis(covid_df):
    """Perform exploratory data analysis and generate visualizations"""
    # Get the latest data for each country
    latest_data = covid_df.sort_values('date').groupby('location').last().reset_index()
    
    # Create visualizations directory if it doesn't exist
    import os
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # 1. Top countries by total cases
    top_cases = latest_data.nlargest(10, 'total_cases')[['location', 'total_cases', 'total_deaths']]
    top_cases['death_rate'] = top_cases['total_deaths'] / top_cases['total_cases']
    
    plt.figure(figsize=(12, 8))
    plt.barh(top_cases['location'], top_cases['total_cases'], color='skyblue')
    plt.xlabel('Total Cases')
    plt.title('Top 10 Countries by Total COVID-19 Cases')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('visualizations/top_countries_cases.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Time series analysis for selected countries
    selected_countries = ['United States', 'India', 'Brazil', 'Germany', 'Kenya', 'South Africa']
    selected_data = covid_df[covid_df['location'].isin(selected_countries)]
    
    # Plot total cases over time
    plt.figure(figsize=(14, 8))
    for country in selected_countries:
        country_data = selected_data[selected_data['location'] == country]
        plt.plot(country_data['date'], country_data['total_cases'], label=country, linewidth=2)

    plt.title('Total COVID-19 Cases Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Total Cases', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('visualizations/cases_over_time.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Vaccination progress
    vaccination_by_continent = latest_data.groupby('continent').agg({
        'people_fully_vaccinated': 'sum',
        'population': 'sum'
    }).reset_index()

    vaccination_by_continent['vaccination_rate'] = vaccination_by_continent['people_fully_vaccinated'] / vaccination_by_continent['population']
    
    plt.figure(figsize=(10, 6))
    plt.bar(vaccination_by_continent['continent'], vaccination_by_continent['vaccination_rate'], color='lightgreen')
    plt.title('Vaccination Rates by Continent')
    plt.ylabel('Proportion Fully Vaccinated')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/vaccination_by_continent.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Correlation heatmap
    correlation_data = latest_data[['total_cases_per_million', 'total_deaths_per_million', 
                                   'population_density', 'median_age', 'aged_65_older',
                                   'gdp_per_capita', 'cardiovasc_death_rate', 
                                   'diabetes_prevalence', 'hospital_beds_per_thousand',
                                   'life_expectancy', 'human_development_index']].dropna()

    corr_matrix = correlation_data.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix of COVID-19 Metrics and Socioeconomic Factors', fontsize=16)
    plt.tight_layout()
    plt.savefig('visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return latest_data

def generate_choropleth_maps(latest_data):
    """Generate interactive choropleth maps"""
    # Cases per million map
    fig = px.choropleth(latest_data, 
                        locations="location",
                        locationmode='country names',
                        color="cases_per_million",
                        hover_name="location",
                        hover_data=["total_cases", "total_deaths", "death_rate"],
                        color_continuous_scale=px.colors.sequential.Plasma,
                        title="COVID-19 Cases per Million People by Country")
    fig.show()
    fig.write_html("visualizations/cases_choropleth.html")
    
    # Vaccination map
    fig = px.choropleth(latest_data, 
                        locations="location",
                        locationmode='country names',
                        color="people_fully_vaccinated",
                        hover_name="location",
                        hover_data=["people_fully_vaccinated", "population", "total_vaccinations"],
                        color_continuous_scale=px.colors.sequential.Viridis,
                        title="People Fully Vaccinated by Country")
    fig.show()
    fig.write_html("visualizations/vaccination_choropleth.html")

def generate_insights_report(latest_data, covid_df):
    """Generate insights report based on the analysis"""
    print("="*60)
    print("COVID-19 GLOBAL DATA ANALYSIS: KEY INSIGHTS")
    print("="*60)
    
    # Global statistics
    total_cases = latest_data['total_cases'].sum()
    total_deaths = latest_data['total_deaths'].sum()
    avg_death_rate = latest_data['death_rate'].mean()
    
    print(f"\n1. GLOBAL IMPACT:")
    print(f"   - Total confirmed cases: {total_cases:,.0f}")
    print(f"   - Total confirmed deaths: {total_deaths:,.0f}")
    print(f"   - Average global death rate: {avg_death_rate:.2%}")
    
    # Vaccination progress
    total_vaccinated = latest_data['people_fully_vaccinated'].sum()
    world_population = latest_data['population'].sum()
    global_vax_rate = total_vaccinated / world_population
    
    print(f"\n2. VACCINATION PROGRESS:")
    print(f"   - Global vaccination rate: {global_vax_rate:.2%}")
    
    # Regional disparities
    vax_by_continent = latest_data.groupby('continent')['people_fully_vaccinated'].sum() / latest_data.groupby('continent')['population'].sum()
    highest_vax = vax_by_continent.idxmax()
    lowest_vax = vax_by_continent.idxmin()
    
    print(f"\n3. REGIONAL DISPARITIES:")
    print(f"   - Highest vaccination rate: {highest_vax} ({vax_by_continent[highest_vax]:.2%})")
    print(f"   - Lowest vaccination rate: {lowest_vax} ({vax_by_continent[lowest_vax]:.2%})")
    
    # Top countries by cases
    top_5_cases = latest_data.nlargest(5, 'total_cases')[['location', 'total_cases', 'total_deaths']]
    print(f"\n4. TOP 5 COUNTRIES BY TOTAL CASES:")
    for i, (_, row) in enumerate(top_5_cases.iterrows(), 1):
        print(f"   {i}. {row['location']}: {row['total_cases']:,.0f} cases, {row['total_deaths']:,.0f} deaths")
    
    # Correlation insights
    correlation_data = latest_data[['total_cases_per_million', 'total_deaths_per_million', 
                                   'population_density', 'median_age', 'gdp_per_capita',
                                   'hospital_beds_per_thousand']].corr()
    
    cases_corr = correlation_data['total_cases_per_million']
    deaths_corr = correlation_data['total_deaths_per_million']
    
    print(f"\n5. KEY CORRELATIONS:")
    print(f"   - Cases per million correlates most strongly with:")
    print(f"     * GDP per capita: {cases_corr['gdp_per_capita']:.3f}")
    print(f"     * Population density: {cases_corr['population_density']:.3f}")
    
    print(f"   - Deaths per million correlates most strongly with:")
    print(f"     * Median age: {deaths_corr['median_age']:.3f}")
    print(f"     * Hospital beds per thousand: {deaths_corr['hospital_beds_per_thousand']:.3f} (negative)")
    
    print(f"\n6. RECOMMENDATIONS:")
    print(f"   - Focus vaccination efforts in regions with low vaccination rates")
    print(f"   - Consider socioeconomic factors in pandemic response planning")
    print(f"   - Strengthen healthcare infrastructure, especially in regions with high death rates")
    
    print("\n" + "="*60)

def main():
    """Main function to run the COVID-19 data analysis"""
    print("COVID-19 Global Data Tracker Analysis")
    print("Loading data...")
    
    # Load the data
    df = load_data()
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Clean the data
    print("Cleaning data...")
    covid_df = clean_data(df)
    
    # Perform EDA and generate visualizations
    print("Performing exploratory data analysis...")
    latest_data = exploratory_data_analysis(covid_df)
    
    # Generate choropleth maps
    print("Generating choropleth maps...")
    generate_choropleth_maps(latest_data)
    
    # Generate insights report
    print("Generating insights report...")
    generate_insights_report(latest_data, covid_df)
    
    print("Analysis complete! Check the 'visualizations' folder for output files.")

if __name__ == "__main__":
    main()