from typing import Union
from world_cup_winner import world_cup_group_match, initialize_world_cup, initialize_world_cup_rankings, \
                            get_actual_result, get_home_win_prob, get_all_matches
from world_cup_winner import world_cup_rankings # 리스트

from fastapi import FastAPI,File,UploadFile, requests, Form, Request
import numpy, pandas, uvicorn
from starlette.responses import HTMLResponse, RedirectResponse

app = FastAPI()


user_capital = 100.0

current_match = None
matches_queue = []
world_cup = initialize_world_cup()
world_cup_rankings = initialize_world_cup_rankings()

def dataframe_to_html(df: pandas.DataFrame) -> str:
    return df.to_html(classes='table table-striped')

# print('check', dataframe_to_html(world_cup_group_match()))

@app.get("/", response_class=HTMLResponse)

async def read_root():
    return HTMLResponse(content="""
        <html>
            <head><title>World Cup Predictor</title></head>
            <body>
                <h1>World Cup Predictor</h1>
                <a href="/start" class="btn btn-primary">Start Prediction</a>
            </body>
        </html>
    """)
    
@app.get("/start", response_class=HTMLResponse)
async def start_prediction():
    global matches_queue, current_match, world_cup, world_cup_rankings, user_capital
    world_cup = initialize_world_cup()
    world_cup_rankings = initialize_world_cup_rankings()
    matches_queue = get_all_matches(world_cup)
    current_match = matches_queue.pop(0)
    user_capital = 100.0
    return RedirectResponse(url="/show-current-match")



@app.get("/show-current-match", response_class=HTMLResponse)
async def show_current_match():
    global current_match, user_capital
    if current_match is None:
        return HTMLResponse(content="<h1>No more matches to predict!</h1><a href='/start' class='btn btn-primary'>Start Over</a>")
    
    home, away = current_match
    home_win_prob = get_home_win_prob(home, away)

    return HTMLResponse(content=f"""
        <html>
            <head><title>Predict Match</title></head>
            <body>
                <h1>Predict Match Outcome</h1>
                <p>{home} (home) {(1/home_win_prob):.2f} vs {away} (away) {(1/(1-home_win_prob)):.2f}</p>
                
                <form action="/predict" method="post">
                    <input type="hidden" name="home" value="{home}">
                    <input type="hidden" name="away" value="{away}">
                    <label>
                        <input type="radio" name="prediction" value="home"> Home Win
                    </label>
                    <label>
                        <input type="radio" name="prediction" value="away"> Away Win
                    </label>
                    <label>
                        <input type="radio" name="prediction" value="draw"> Draw
                    </label>
                    <br>
                    <label>
                        Betting Amount: <input type="number" name="betting_amount" min="1" max="{user_capital}" step="0.01" required>
                    </label>
                    <button type="submit">Submit Prediction</button>
                </form>
                <p>Current Capital: ${user_capital:.2f}</p>
            </body>
        </html>
    """)


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, home: str = Form(...), away: str = Form(...), prediction: str = Form(...), betting_amount: float = Form(...)):
    global current_match, matches_queue, user_capital
    home_win_prob = get_home_win_prob(home, away)
    actual_result = get_actual_result(home_win_prob)
    
    result_message = "Prediction Failed"
    payout = 0.0
    if (prediction == "home" and actual_result == "home"):
        payout = round(betting_amount * (1/home_win_prob), 2)
        result_message = f"Prediction Successful! You won ${payout:.2f}"
    elif (prediction == "away" and actual_result == "away"):
        payout = round(betting_amount * (1/(1-home_win_prob)), 2)
        result_message = f"Prediction Successful! You won ${payout:.2f}"
    elif (prediction == "draw" and actual_result == "draw"):
        payout = round(betting_amount * 2, 2)  # Assuming draw payout is double the betting amount
        result_message = f"Prediction Successful! You won ${payout:.2f}"
        
    user_capital = round(user_capital + payout - betting_amount, 2)
    if user_capital <= 0:
        user_capital = 0
        return HTMLResponse(content=f"""
            <html>
                <head><title>Game Over</title></head>
                <body>
                    <h1>Game Over! You've run out of capital.</h1>
                    <a href='/start' class='btn btn-primary'>Start Over</a>
                </body>
            </html>
        """)
    
    next_match_link = ""
    if matches_queue:
        current_match = matches_queue.pop(0)
        next_match_link = "<a href='/show-current-match' class='btn btn-primary'>Next Match</a>"
    else:
        current_match = None
        next_match_link = "<a href='/start' class='btn btn-primary'>Start Over</a>"

    return HTMLResponse(content=f"""
        <html>
            <head><title>Prediction Result</title></head>
            <body>
                <h1>{result_message}</h1>
                <p>Current Capital: ${user_capital:.2f}</p>
                {next_match_link}
            </body>
        </html>
    """)


@app.get("/show-data", response_class=HTMLResponse)
async def show_data():
    df_html = dataframe_to_html(world_cup_group_match())
    
    html_content = f"""
    <html>
        <head>
            <title>World Cup Data</title>
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
            <style>
                .table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                .table th, .table td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                }}
                .table th {{
                    background-color: #f4f4f4;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>World Cup Data</h1>
                {df_html}
                <a href="/" class="btn btn-primary">Go Back</a>
                <a href="/show-data/elimination-rounds" class="btn btn-secondary">Show Elimination Rounds Data</a>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

