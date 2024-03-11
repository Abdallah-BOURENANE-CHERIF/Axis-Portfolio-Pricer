import streamlit as st
import numpy as np
from scipy.stats import norm
import streamlit.components.v1 as components
from jinja2 import Template


###########################################################################################
###########################################################################################
################################ MONTE - CARLO ############################################
###########################################################################################
##########################################################################################


def monte_carlo_option_pricer(S, K, T, r, q, sigma, option_type='call', num_simulations=100000):
    dt = T / 252  # Assume 252 trading days in a year
    np.random.seed(42)  # For reproducibility
    
    # Generate random normal samples
    z = np.random.normal(0, 1, (num_simulations, 1))
    
    # Calculate simulated stock prices using geometric Brownian motion
    ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    
    # Calculate option payoff
    if option_type == 'call':
        payoff = np.maximum(ST - K, 0)
    elif option_type == 'put':
        payoff = np.maximum(K - ST, 0)
    else:
        raise ValueError("Type d'option non valide. Utilisez 'call' ou 'put'.")
    
    # Discounted expected payoff to get option price
    option_price = np.exp(-r * T) * np.mean(payoff)
    
    return option_price


def monte_carlo_greeks(S, K, T, r, q, sigma, option_type='call', num_simulations=100000, epsilon=0.01):
    # Calculate delta using finite difference method
    delta_S = S * epsilon
    price_up = monte_carlo_option_pricer(S + delta_S, K, T, r, q, sigma, option_type, num_simulations)
    price_down = monte_carlo_option_pricer(S - delta_S, K, T, r, q, sigma, option_type, num_simulations)
    delta = (price_up - price_down) / (2 * delta_S)
    
    # Calculate gamma using finite difference method
    gamma = (price_up - 2 * monte_carlo_option_pricer(S, K, T, r, q, sigma, option_type, num_simulations) + price_down) / (delta_S**2)
    
    # Calculate vega using finite difference method
    delta_sigma = sigma * epsilon
    price_up = monte_carlo_option_pricer(S, K, T, r, q, sigma + delta_sigma, option_type, num_simulations)
    price_down = monte_carlo_option_pricer(S, K, T, r, q, sigma - delta_sigma, option_type, num_simulations)
    vega = (price_up - price_down) / (2 * delta_sigma)
    
    # Calculate theta using finite difference method
    delta_T = T * epsilon
    price_up = monte_carlo_option_pricer(S, K, T - delta_T, r, q, sigma, option_type, num_simulations)
    price_down = monte_carlo_option_pricer(S, K, T + delta_T, r, q, sigma, option_type, num_simulations)
    theta = (price_down - price_up) / (2 * delta_T)
    
    return delta, gamma, vega, theta





def portfolio_option_pricer_MT(options,num_simulations):
    
    total_price = 0
    total_delta = 0
    total_gamma = 0
    total_vega = 0
    total_theta = 0

    for option in options:
        option_price = monte_carlo_option_pricer(option['S'], option['K'], option['T'],
                                                   option['r'], option['q'], option['sigma'], option['type'],num_simulations)
        total_price += option_price * option['quantity']

        delta, gamma, vega, theta = monte_carlo_greeks(option['S'], option['K'], option['T'],
                                                  option['r'], option['q'], option['sigma'], option['type'],num_simulations,epsilon=0.01)

        total_delta += delta * option['quantity']
        total_gamma += gamma * option['quantity']
        total_vega += vega * option['quantity']
        total_theta += theta * option['quantity']

    return total_price, total_delta, total_gamma, total_vega, total_theta

##########################################################################################
##########################################################################################
################################ BLACK - SCHOLES #########################################
##########################################################################################
##########################################################################################



def black_scholes_option_pricer(S, K, T, r, q, sigma, option_type='call'):
   
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        option_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("Type d'option non valide. Utilisez 'call' ou 'put'.")

    return option_price


def option_greeks(S, K, T, r, q, sigma, option_type):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    N_prime_d1 = norm.pdf(d1)

    delta = 0
    gamma = 0
    vega = 0
    theta = 0

    if option_type == 'call':
        delta = norm.cdf(d1)
        gamma = N_prime_d1 / (S * sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * N_prime_d1
        theta = -(S * sigma * N_prime_d1) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        delta = norm.cdf(d1) - 1
        gamma = N_prime_d1 / (S * sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * N_prime_d1
        theta = -(S * sigma * N_prime_d1) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)

    return delta, gamma, vega, theta



def portfolio_option_pricer_BC(options):
    total_price = 0
    total_delta = 0
    total_gamma = 0
    total_vega = 0
    total_theta = 0

    for option in options:
        option_price = black_scholes_option_pricer(option['S'], option['K'], option['T'],
                                                   option['r'], option['q'], option['sigma'], option['type'])
        total_price += option_price * option['quantity']

        delta, gamma, vega, theta = option_greeks(option['S'], option['K'], option['T'],
                                                  option['r'], option['q'], option['sigma'], option['type'])

        total_delta += delta * option['quantity']
        total_gamma += gamma * option['quantity']
        total_vega += vega * option['quantity']
        total_theta += theta * option['quantity']

    return total_price, total_delta, total_gamma, total_vega, total_theta



##########################################################################################
##########################################################################################
################################ METRICS AND STYLERS  ####################################
##########################################################################################
##########################################################################################



def _build_metric(label, value):
    html_text = """
    <style>
    .metric {
       font-family: "IBM Plex Sans", sans-serif;
       text-align: center;
    }
    .metric .value {
       font-size: 48px;
       line-height: 1.6;
    }
    .metric .label {
       letter-spacing: 2px;
       font-size: 14px;
       text-transform: uppercase;
    }

    </style>
    <div class="metric">
       <div class="value">
          {{ value }}
       </div>
       <div class="label">
          {{ label }}
       </div>
    </div>
    """
    html = Template(html_text)
    return html.render(label=label, value=value)

def metric_row(data):
    columns = st.beta_columns(len(data))
    for i, (label, value) in enumerate(data.items()):
        with columns[i]:
            components.html(_build_metric(label, value))

def metric(label, value):
    components.html(_build_metric(label, value))





def style_button_row(clicked_button_ix, n_buttons):
    def get_button_indices(button_ix):
        return {
            'nth_child': button_ix,
            'nth_last_child': n_buttons - button_ix + 1
        }

    clicked_style = """
    div[data-testid*="stHorizontalBlock"] > div:nth-child(%(nth_child)s):nth-last-child(%(nth_last_child)s) button {
        border-color: rgb(255, 75, 75);
        color: rgb(255, 75, 75);
        box-shadow: rgba(255, 75, 75, 0.5) 0px 0px 0px 0.2rem;
        outline: currentcolor none medium;
    }
    """
    unclicked_style = """
    div[data-testid*="stHorizontalBlock"] > div:nth-child(%(nth_child)s):nth-last-child(%(nth_last_child)s) button {
        pointer-events: none;
        cursor: not-allowed;
        opacity: 0.65;
        filter: alpha(opacity=65);
        -webkit-box-shadow: none;
        box-shadow: none;
    }
    """
    style = ""
    for ix in range(n_buttons):
        ix += 1
        if ix == clicked_button_ix:
            style += clicked_style % get_button_indices(ix)
        else:
            style += unclicked_style % get_button_indices(ix)
    st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)


st.set_page_config(layout="wide")
# Sidebar
st.sidebar.image("Axis logo.png", use_column_width=True)
#st.sidebar.title("SELECT THE OPTIONS:")


# Main content area
st.title("AXIS OPTIONS PORTFOLIO PRICER")
#st.header('This is the First Axis App Yes !! ' , divider='rainbow')



foot = f"""
<div style="
    position: fixed;
    bottom: 0;
    left: 30%;
    right: 0;
    width: 50%;
    padding: 0px 0px;
    text-align: center;
">
    <p> Made by Abdallah BOURENANE CHERIF (Beta Version)</a></p>
</div>
"""



st.markdown(foot, unsafe_allow_html=True)



##########################################################################################
##########################################################################################
################################ IMPLEMENTATION OF THE INTERFACE #########################
##########################################################################################
##########################################################################################





if 'active_sets' not in st.session_state:
    st.session_state['active_sets'] = [1]
if 'options_portfolio' not in st.session_state:
    st.session_state['options_portfolio'] = []
    


def create_input_set(set_number):
    col_widths = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,0.5]  # Adjusted for delete button
    cols = st.columns(col_widths)
    
    option_dict = {}
    
    with cols[0]:
        option_dict['S'] = st.slider(f'S {set_number}:', 0, 200, 100, key=f'S_{set_number}')
    
    with cols[1]:
        option_dict['r'] = st.number_input(f'R {set_number}:', key=f'R_{set_number}')
    
    with cols[2]:
        option_dict['T'] = st.number_input(f'T {set_number}:', key=f'T_{set_number}')
    
    with cols[3]:
        option_dict['K'] = st.slider(f'K {set_number}:', 50, 160, 100, key=f'K_{set_number}')
    
    with cols[4]:
        option_dict['q'] = st.number_input(f'Q {set_number}:', min_value=0.0, max_value=1.0, value=0.02, step=0.01, key=f'q_{set_number}')
    
    with cols[5]:
        option_dict['sigma'] = st.number_input(f'SIGMA {set_number}:', min_value=0.0, max_value=1.0, value=0.02, step=0.01, key=f'SIGMA_{set_number}')
    
    with cols[6]:
        option_dict['type'] = st.selectbox(f'Option {set_number}:', ['call', 'put'], key=f'Option_{set_number}')
    
    with cols[7]:
        option_dict['quantity'] = st.number_input(f'Quantity {set_number}:', min_value=0, max_value=100, value=1, step=1, key=f'Quantity_{set_number}')
    
    with cols[8]:  # Add delete button in the last column
        if st.button("❌", key=f'delete_{set_number}'):
            st.session_state['active_sets'].remove(set_number)
            st.experimental_rerun()
    
    return option_dict  # Return the created option dictionary

# Initialize list to store all option dictionaries
all_options = []

# Loop to generate the input sets based on the active sets
for set_number in st.session_state['active_sets']:
    option_dict = create_input_set(set_number)
    all_options.append(option_dict)



# Button to add another set of inputs
if st.button('Add an option ➕'):
    # Add new set by incrementing the highest set number by 1
    new_set_number = max(st.session_state['active_sets']) + 1 if st.session_state['active_sets'] else 1
    st.session_state['active_sets'].append(new_set_number)
    st.experimental_rerun()



Model = st.sidebar.selectbox(
    'CHOOSE THE PRICING MODEL:',
    ('ALL', 'Black-Scholes', 'Monte-Carlo'))



def local_css():
    st.markdown("""
        <style>
        div.stButton > button {
            color: white;
            background-color: green;
            border: none;
            width: 100%;
            padding: 10px 24px;
            border-radius: 4px;
            margin-bottom: 5px;
        }
        </style>
        """, unsafe_allow_html=True)

# Call the function to apply the CSS styles
local_css()

# Spacer to push the button to the bottom of the sidebar
for _ in range(30):  # Adjust the range number based on your content to push the button to the bottom
    st.sidebar.write("")


    
# Create the button with custom style at the bottom of the sidebar
if st.sidebar.button('RUN', key='run_button'):
    st.sidebar.write('YOUR MODELS HAVE BEEN EXECUTED')
    total_portfolio_price_BC, total_portfolio_delta_BC, total_portfolio_gamma_BC, total_portfolio_vega_BC, total_portfolio_theta_BC = portfolio_option_pricer_BC(all_options)
    
    
    col0,col1, col2, col3, col4 , col5 = st.columns(6)
   
    
    if Model == 'Black-Scholes':
    

        col0.write("Black-Scholes")
        col1.metric("TOTAL PORTFOLIO PRICE",  "{:.2f}".format(total_portfolio_price_BC))
        col2.metric("TOTAL DELTA DE PORTFOLIO",  "{:.2f}".format(total_portfolio_delta_BC))
        col3.metric("TOTAL GAMMA DE PORTFOLIO",  "{:.2f}".format(total_portfolio_gamma_BC))
        col4.metric("TOTAL VEGA PORTFOLIIO ",  "{:.2f}".format(total_portfolio_vega_BC))
        col5.metric("TOTAL THETA PORTFOLIO ",  "{:.2f}".format(total_portfolio_theta_BC))
        
    
    if Model == 'Monte-Carlo':
            
        #num_simulations = st.sidebar.slider(f'Number_of_simulations {set_number}:', 0, 200000, 100000, key=f'Number_of_simulations_{set_number}')
                
        #num_simulations = st.number_input(f'Number_of_simulations {set_number}:', key=f'Number_of_simulations_{set_number}') 
                
                
        total_portfolio_price_MT, total_portfolio_delta_MT, total_portfolio_gamma_MT, total_portfolio_vega_MT, total_portfolio_theta_MT = portfolio_option_pricer_MT(all_options,num_simulations = 100000) 
            
        
        col0.write("Monte-Carlo")
        col1.metric("TOTAL PORTFOLIO PRICE",  "{:.2f}".format(total_portfolio_price_MT))
        col2.metric("TOTAL DELTA DE PORTFOLIO",  "{:.2f}".format(total_portfolio_delta_MT))
        col3.metric("TOTAL GAMMA DE PORTFOLIO",  "{:.2f}".format(total_portfolio_gamma_MT))
        col4.metric("TOTAL VEGA PORTFOLIIO ",  "{:.2f}".format(total_portfolio_vega_MT))
        col5.metric("TOTAL THETA PORTFOLIO ",  "{:.2f}".format(total_portfolio_theta_MT))
    
        
    if Model == 'ALL':
        
        col0,col1, col2, col3, col4 , col5 = st.columns(6)
        coll0,coll1, coll2, coll3, coll4 , coll5 = st.columns(6)
        
        col0.write("Black-Scholes")
        col1.metric("TOTAL PORTFOLIO PRICE",  "{:.2f}".format(total_portfolio_price_BC))
        col2.metric("TOTAL DELTA DE PORTFOLIO",  "{:.2f}".format(total_portfolio_delta_BC))
        col3.metric("TOTAL GAMMA DE PORTFOLIO",  "{:.2f}".format(total_portfolio_gamma_BC))
        col4.metric("TOTAL VEGA PORTFOLIIO ",  "{:.2f}".format(total_portfolio_vega_BC))
        col5.metric("TOTAL THETA PORTFOLIO ",  "{:.2f}".format(total_portfolio_theta_BC))
        
        total_portfolio_price_MT, total_portfolio_delta_MT, total_portfolio_gamma_MT, total_portfolio_vega_MT, total_portfolio_theta_MT = portfolio_option_pricer_MT(all_options,num_simulations = 100000) 
            
        
        coll0.write("Monte-Carlo")
        coll1.metric("TOTAL PORTFOLIO PRICE",  "{:.2f}".format(total_portfolio_price_MT))
        coll2.metric("TOTAL DELTA DE PORTFOLIO",  "{:.2f}".format(total_portfolio_delta_MT))
        coll3.metric("TOTAL GAMMA DE PORTFOLIO",  "{:.2f}".format(total_portfolio_gamma_MT))
        coll4.metric("TOTAL VEGA PORTFOLIIO ",  "{:.2f}".format(total_portfolio_vega_MT))
        coll5.metric("TOTAL THETA PORTFOLIO ",  "{:.2f}".format(total_portfolio_theta_MT))
    









