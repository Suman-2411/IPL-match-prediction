# IPL Match Winner Predictor

🔗 **Live app:** [huggingface.co/spaces/Suman-24/ipl-match-predictor](https://huggingface.co/spaces/Suman-24/ipl-match-predictor)

## What this project is about

Cricket, and the IPL in particular, is full of momentum swings and "gut feeling" predictions — pundits and fans are always guessing who's going to win before a ball is even bowled. I wanted to see how far you could get with an actual data-driven approach: given two teams and a venue, can a model make a genuinely informed prediction about who's more likely to win?

This project builds a match-winner predictor trained on historical IPL data, wrapped in a simple Streamlit app where you pick the two teams and the venue and get a prediction back instantly.

## How it works

At its core, this is a classification problem: given the details of an upcoming match, predict which team wins. The tricky part isn't the model — it's the features. Raw team names and venues don't tell you much on their own, so I engineered features that actually capture cricketing context:

- **Team form** — how well each team has been performing recently, rather than just their overall historical record
- **Head-to-head rate** — how these two specific teams have historically fared against each other
- **Venue win rate** — how much of an edge (or disadvantage) a team has historically had at that particular ground

Adding these engineered features nudged accuracy up from around 52% to about 53%. That might not sound like a huge jump, but in a sport as famously unpredictable as T20 cricket — where a single over can flip a match — even a modest, honest improvement over a coin-flip baseline is meaningful. It's also a good reminder that domain-informed features often matter more than throwing a fancier model at the problem.

## Tech stack

- **Model:** XGBoost classifier
- **App/UI:** Streamlit
- **Deployment:** Hugging Face Spaces

## Why I built this

This is part of my portfolio as I move into data science and machine learning roles. Sports prediction is a great sandbox for applied ML — the data is messy and real-world, the baseline (random guessing / always picking the favorite) is easy to beat but hard to meaningfully improve on, and it forces you to think carefully about feature engineering rather than just model selection.

## Try it yourself

Head over to the [live Space](https://huggingface.co/spaces/Suman-24/ipl-match-predictor), pick two IPL teams and a venue, and see who the model favors.

## Running it locally

If you'd rather run it on your own machine instead of using the hosted Space:

```bash
# Clone the repo
git clone https://github.com/Suman-2411/ipl-match-predictor.git
cd ipl-match-predictor

# Install dependencies
pip install -r requirements.txt
```

Then start the app:

```bash
streamlit run app.py
```

The app will open automatically in your browser (usually at `http://localhost:8501`).

> **Note:** update the commands above if your entry point or dependency file names differ — swap `app.py`/`requirements.txt` for whatever your repo actually uses.

## Author

**Suman Thangadurai**
- GitHub: [Suman-2411](https://github.com/Suman-2411)
- LinkedIn: [suman-thangadurai](https://linkedin.com/in/suman-thangadurai)
