from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import streamlit as st


# =========================
# Basic settings
# =========================

st.set_page_config(
    page_title="2026 世界杯预测",
    page_icon="⚽",
    layout="wide",
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"


# =========================
# Load model and data
# =========================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_DIR / "match_result_model_elo.pkl")
    model_features = joblib.load(MODEL_DIR / "model_features_elo.pkl")
    return model, model_features


@st.cache_data
def load_data():
    latest_team_stats = pd.read_csv(DATA_DIR / "latest_team_stats_elo.csv")
    world_cup_stats = pd.read_csv(DATA_DIR / "world_cup_2026_stats_elo.csv")
    squad_strength = pd.read_csv(DATA_DIR / "world_cup_2026_squad_strength_scored.csv")

    injury_pilot_path = DATA_DIR / "world_cup_2026_squad_strength_scored_with_injury_pilot.csv"

    if injury_pilot_path.exists():
        squad_strength_injury = pd.read_csv(injury_pilot_path)
    else:
        squad_strength_injury = squad_strength.copy()
        squad_strength_injury["injury_adjusted_squad_strength_score"] = (
            squad_strength_injury["squad_strength_score"]
        )
        squad_strength_injury["injury_data_quality"] = "injury_file_not_found_fallback"

    return latest_team_stats, world_cup_stats, squad_strength, squad_strength_injury


model, model_features = load_model()
latest_team_stats, world_cup_stats, squad_strength, squad_strength_injury = load_data()

display_to_dataset = dict(
    zip(
        world_cup_stats["team_display"],
        world_cup_stats["team"],
    )
)

ROLLING_FEATURES = [
    "goals_for_roll10",
    "goals_against_roll10",
    "points_roll10",
    "win_rate_roll10",
    "draw_rate_roll10",
    "loss_rate_roll10",
]


# =========================
# Team list and groups
# =========================

GROUPS_2026 = {
    "Group A": ["Mexico", "South Africa", "Korea Republic", "Czechia"],
    "Group B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "Group C": ["Brazil", "Haiti", "Scotland", "Morocco"],
    "Group D": ["United States", "Australia", "Türkiye", "Paraguay"],
    "Group E": ["Germany", "Côte d'Ivoire", "Ecuador", "Curaçao"],
    "Group F": ["Netherlands", "Sweden", "Tunisia", "Japan"],
    "Group G": ["Belgium", "IR Iran", "New Zealand", "Egypt"],
    "Group H": ["Uruguay", "Cabo Verde", "Spain", "Saudi Arabia"],
    "Group I": ["Norway", "France", "Senegal", "Iraq"],
    "Group J": ["Argentina", "Austria", "Jordan", "Algeria"],
    "Group K": ["Portugal", "Uzbekistan", "Colombia", "Congo DR"],
    "Group L": ["England", "Ghana", "Panama", "Croatia"],
}

GROUP_FIXTURE_PATTERN = [
    (0, 1),
    (2, 3),
    (0, 2),
    (3, 1),
    (3, 0),
    (1, 2),
]

TEAM_OPTIONS = list(world_cup_stats["team_display"])


# =========================
# Prediction functions
# =========================

def get_dataset_team_name(team_name):
    return display_to_dataset.get(team_name, team_name)


def get_team_stats(team_name):
    dataset_name = get_dataset_team_name(team_name)

    stats = latest_team_stats[
        latest_team_stats["team"] == dataset_name
    ]

    if stats.empty:
        raise ValueError(f"Cannot find team data: {team_name} / {dataset_name}")

    return stats.iloc[0]


def build_match_features_elo(
    team_a,
    team_b,
    neutral=True,
    is_world_cup=True,
):
    team_a_stats = get_team_stats(team_a)
    team_b_stats = get_team_stats(team_b)

    row = {}

    row["elo_diff"] = (
        team_a_stats["elo_rating"] - team_b_stats["elo_rating"]
    )

    for col in ROLLING_FEATURES:
        row[f"{col}_diff"] = (
            team_a_stats[col] - team_b_stats[col]
        )

    row["neutral"] = int(neutral)
    row["is_world_cup"] = int(is_world_cup)

    features = pd.DataFrame([row])
    features = features[model_features]

    return features


def predict_match_elo_raw(
    team_a,
    team_b,
    neutral=True,
    is_world_cup=True,
):
    features = build_match_features_elo(
        team_a=team_a,
        team_b=team_b,
        neutral=neutral,
        is_world_cup=is_world_cup,
    )

    proba = model.predict_proba(features)[0]

    result = pd.DataFrame(
        {
            "result": [
                f"{team_a} win",
                "Draw",
                f"{team_b} win",
            ],
            "probability": proba,
        }
    )

    result["probability_percent"] = (
        result["probability"] * 100
    ).round(1)

    return result, features


def predict_match_elo(
    team_a,
    team_b,
    neutral=True,
    is_world_cup=True,
):
    result_ab, features_ab = predict_match_elo_raw(
        team_a=team_a,
        team_b=team_b,
        neutral=neutral,
        is_world_cup=is_world_cup,
    )

    result_ba, features_ba = predict_match_elo_raw(
        team_a=team_b,
        team_b=team_a,
        neutral=neutral,
        is_world_cup=is_world_cup,
    )

    p_a_win = (
        result_ab.loc[0, "probability"]
        + result_ba.loc[2, "probability"]
    ) / 2

    p_draw = (
        result_ab.loc[1, "probability"]
        + result_ba.loc[1, "probability"]
    ) / 2

    p_b_win = (
        result_ab.loc[2, "probability"]
        + result_ba.loc[0, "probability"]
    ) / 2

    result = pd.DataFrame(
        {
            "result": [
                f"{team_a} win",
                "Draw",
                f"{team_b} win",
            ],
            "probability": [
                p_a_win,
                p_draw,
                p_b_win,
            ],
        }
    )

    result["probability_percent"] = (
        result["probability"] * 100
    ).round(1)

    return result, features_ab


def make_team_comparison(team_a, team_b):
    team_a_stats = get_team_stats(team_a)
    team_b_stats = get_team_stats(team_b)

    comparison = pd.DataFrame(
        [
            {
                "team": team_a,
                "elo_rating": team_a_stats["elo_rating"],
                "goals_for_roll10": team_a_stats["goals_for_roll10"],
                "goals_against_roll10": team_a_stats["goals_against_roll10"],
                "points_roll10": team_a_stats["points_roll10"],
                "win_rate_roll10": team_a_stats["win_rate_roll10"],
                "draw_rate_roll10": team_a_stats["draw_rate_roll10"],
                "loss_rate_roll10": team_a_stats["loss_rate_roll10"],
                "last_match_date": team_a_stats["last_match_date"],
            },
            {
                "team": team_b,
                "elo_rating": team_b_stats["elo_rating"],
                "goals_for_roll10": team_b_stats["goals_for_roll10"],
                "goals_against_roll10": team_b_stats["goals_against_roll10"],
                "points_roll10": team_b_stats["points_roll10"],
                "win_rate_roll10": team_b_stats["win_rate_roll10"],
                "draw_rate_roll10": team_b_stats["draw_rate_roll10"],
                "loss_rate_roll10": team_b_stats["loss_rate_roll10"],
                "last_match_date": team_b_stats["last_match_date"],
            },
        ]
    )

    numeric_cols = [
        "elo_rating",
        "goals_for_roll10",
        "goals_against_roll10",
        "points_roll10",
        "win_rate_roll10",
        "draw_rate_roll10",
        "loss_rate_roll10",
    ]

    comparison[numeric_cols] = comparison[numeric_cols].round(3)

    return comparison
def get_active_squad_strength_table(use_injury_adjustment=False):
    if (
        use_injury_adjustment
        and "injury_adjusted_squad_strength_score" in squad_strength_injury.columns
    ):
        return squad_strength_injury

    return squad_strength


def get_squad_strength_score(team_name, use_injury_adjustment=False):
    active_squad_strength = get_active_squad_strength_table(
        use_injury_adjustment=use_injury_adjustment
    )

    row = active_squad_strength[
        active_squad_strength["team_display"] == team_name
    ]

    if row.empty:
        return 0.0

    if (
        use_injury_adjustment
        and "injury_adjusted_squad_strength_score" in active_squad_strength.columns
    ):
        strength_col = "injury_adjusted_squad_strength_score"
    else:
        strength_col = "squad_strength_score"

    return float(row.iloc[0][strength_col])


def make_squad_strength_comparison(team_a, team_b):
    preferred_cols = [
        "team_display",
        "fifa_rank",
        "fifa_points",
        "official_fifa_strength_score",
        "squad_strength_score",
        "squad_strength_method",
        "fifa_confederation",
        "fifa_ranking_movement",
        "fifa_rated_matches",
        "squad_value_m",
        "avg_player_rating",
        "top5_player_rating",
        "star_rating",
        "star_available",
        "injury_count",
        "data_quality",
    ]

    available_cols = [
        col for col in preferred_cols
        if col in squad_strength.columns
    ]

    comparison = squad_strength[
        squad_strength["team_display"].isin([team_a, team_b])
    ][available_cols].copy()

    order_map = {
        team_a: 0,
        team_b: 1,
    }

    comparison["_display_order"] = comparison["team_display"].map(order_map)
    comparison = comparison.sort_values("_display_order")
    comparison = comparison.drop(columns=["_display_order"])

    numeric_cols = [
        "fifa_rank",
        "fifa_points",
        "official_fifa_strength_score",
        "squad_strength_score",
        "fifa_ranking_movement",
        "fifa_rated_matches",
        "squad_value_m",
        "avg_player_rating",
        "top5_player_rating",
        "star_rating",
        "star_available",
        "injury_count",
    ]

    numeric_cols = [
        col for col in numeric_cols
        if col in comparison.columns
    ]

    comparison[numeric_cols] = comparison[numeric_cols].round(3)

    return comparison

def adjust_prediction_with_squad_strength(
    base_prediction,
    team_a,
    team_b,
    adjustment_weight=0.25,
    use_injury_adjustment=False,
):
    adjusted = base_prediction.copy()

    team_a_strength = get_squad_strength_score(
        team_a,
        use_injury_adjustment=use_injury_adjustment,
    )
    team_b_strength = get_squad_strength_score(
        team_b,
        use_injury_adjustment=use_injury_adjustment,
    )

    strength_diff = team_a_strength - team_b_strength

    p_a = adjusted.loc[0, "probability"]
    p_draw = adjusted.loc[1, "probability"]
    p_b = adjusted.loc[2, "probability"]

    win_mass = p_a + p_b

    if p_a <= 0 or p_b <= 0:
        adjusted["adjusted_probability"] = adjusted["probability"]
    else:
        base_odds = p_a / p_b

        adjusted_odds = base_odds * np.exp(
            adjustment_weight * strength_diff
        )

        p_a_adjusted = win_mass * adjusted_odds / (1 + adjusted_odds)
        p_b_adjusted = win_mass / (1 + adjusted_odds)

        adjusted.loc[0, "adjusted_probability"] = p_a_adjusted
        adjusted.loc[1, "adjusted_probability"] = p_draw
        adjusted.loc[2, "adjusted_probability"] = p_b_adjusted

    adjusted["adjusted_probability_percent"] = (
        adjusted["adjusted_probability"] * 100
    ).round(1)

    adjusted["squad_strength_team_a"] = team_a_strength
    adjusted["squad_strength_team_b"] = team_b_strength
    adjusted["squad_strength_diff"] = strength_diff

    return adjusted

def make_probability_chart(prediction):
    fig = px.bar(
        prediction,
        x="result",
        y="probability_percent",
        text="probability_percent",
        title="胜 / 平 / 负概率",
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )

    fig.update_layout(
        yaxis_title="概率 (%)",
        xaxis_title="结果",
        yaxis_range=[0, 100],
    )

    return fig


# =========================
# Group functions
# =========================

def make_group_fixtures(group_name):
    teams = GROUPS_2026[group_name]

    fixtures = []

    for match_no, pair in enumerate(GROUP_FIXTURE_PATTERN, start=1):
        i, j = pair
        fixtures.append(
            {
                "group": group_name,
                "match_no": match_no,
                "team_a": teams[i],
                "team_b": teams[j],
            }
        )

    return pd.DataFrame(fixtures)


def predict_fixture_row(team_a, team_b):
    prediction, _ = predict_match_elo(
        team_a=team_a,
        team_b=team_b,
        neutral=True,
        is_world_cup=True,
    )

    p_a_win = prediction.loc[0, "probability"]
    p_draw = prediction.loc[1, "probability"]
    p_b_win = prediction.loc[2, "probability"]

    return {
        "team_a": team_a,
        "team_b": team_b,
        "p_team_a_win": p_a_win,
        "p_draw": p_draw,
        "p_team_b_win": p_b_win,
        "team_a_win_%": round(p_a_win * 100, 1),
        "draw_%": round(p_draw * 100, 1),
        "team_b_win_%": round(p_b_win * 100, 1),
    }


def predict_group_matches(group_name):
    fixtures = make_group_fixtures(group_name)

    rows = []

    for _, row in fixtures.iterrows():
        pred = predict_fixture_row(row["team_a"], row["team_b"])
        pred["group"] = group_name
        pred["match_no"] = row["match_no"]
        rows.append(pred)

    result = pd.DataFrame(rows)

    return result[
        [
            "group",
            "match_no",
            "team_a",
            "team_b",
            "team_a_win_%",
            "draw_%",
            "team_b_win_%",
            "p_team_a_win",
            "p_draw",
            "p_team_b_win",
        ]
    ]


def make_expected_group_table(group_name):
    teams = GROUPS_2026[group_name]

    table = {
        team: {
            "team": team,
            "expected_points": 0.0,
            "expected_wins": 0.0,
            "expected_draws": 0.0,
            "expected_losses": 0.0,
        }
        for team in teams
    }

    match_predictions = predict_group_matches(group_name)

    for _, row in match_predictions.iterrows():
        team_a = row["team_a"]
        team_b = row["team_b"]

        p_a_win = row["p_team_a_win"]
        p_draw = row["p_draw"]
        p_b_win = row["p_team_b_win"]

        table[team_a]["expected_points"] += 3 * p_a_win + p_draw
        table[team_b]["expected_points"] += 3 * p_b_win + p_draw

        table[team_a]["expected_wins"] += p_a_win
        table[team_b]["expected_wins"] += p_b_win

        table[team_a]["expected_draws"] += p_draw
        table[team_b]["expected_draws"] += p_draw

        table[team_a]["expected_losses"] += p_b_win
        table[team_b]["expected_losses"] += p_a_win

    group_table = pd.DataFrame(table.values())

    group_table["elo_rating"] = group_table["team"].apply(
        lambda x: get_team_stats(x)["elo_rating"]
    )

    group_table = group_table.sort_values(
        ["expected_points", "expected_wins", "elo_rating"],
        ascending=False,
    ).reset_index(drop=True)

    group_table["expected_rank"] = group_table.index + 1

    display_cols = [
        "expected_rank",
        "team",
        "elo_rating",
        "expected_points",
        "expected_wins",
        "expected_draws",
        "expected_losses",
    ]

    group_table = group_table[display_cols]

    numeric_cols = [
        "elo_rating",
        "expected_points",
        "expected_wins",
        "expected_draws",
        "expected_losses",
    ]

    group_table[numeric_cols] = group_table[numeric_cols].round(3)

    return group_table
    
def make_squad_adjusted_group_table(
    group_name,
    adjustment_weight=0.25,
    use_injury_adjustment=False,
):
    teams = GROUPS_2026[group_name]

    table = {
        team: {
            "team": team,
            "squad_adjusted_expected_points": 0.0,
            "squad_adjusted_expected_wins": 0.0,
            "squad_adjusted_expected_draws": 0.0,
            "squad_adjusted_expected_losses": 0.0,
        }
        for team in teams
    }

    fixtures = make_group_fixtures(group_name)

    rows = []

    for _, row in fixtures.iterrows():
        team_a = row["team_a"]
        team_b = row["team_b"]

        base_prediction, _ = predict_match_elo(
            team_a=team_a,
            team_b=team_b,
            neutral=True,
            is_world_cup=True,
        )

        adjusted_prediction = adjust_prediction_with_squad_strength(
            base_prediction=base_prediction,
            team_a=team_a,
            team_b=team_b,
            adjustment_weight=adjustment_weight,
            use_injury_adjustment=use_injury_adjustment,
        )

        p_a_win = adjusted_prediction.loc[0, "adjusted_probability"]
        p_draw = adjusted_prediction.loc[1, "adjusted_probability"]
        p_b_win = adjusted_prediction.loc[2, "adjusted_probability"]

        table[team_a]["squad_adjusted_expected_points"] += 3 * p_a_win + p_draw
        table[team_b]["squad_adjusted_expected_points"] += 3 * p_b_win + p_draw

        table[team_a]["squad_adjusted_expected_wins"] += p_a_win
        table[team_b]["squad_adjusted_expected_wins"] += p_b_win

        table[team_a]["squad_adjusted_expected_draws"] += p_draw
        table[team_b]["squad_adjusted_expected_draws"] += p_draw

        table[team_a]["squad_adjusted_expected_losses"] += p_b_win
        table[team_b]["squad_adjusted_expected_losses"] += p_a_win

        rows.append(
            {
                "match_no": row["match_no"],
                "team_a": team_a,
                "team_b": team_b,
                "team_a_win_%": round(p_a_win * 100, 1),
                "draw_%": round(p_draw * 100, 1),
                "team_b_win_%": round(p_b_win * 100, 1),
            }
        )

    adjusted_matches = pd.DataFrame(rows)

    adjusted_group_table = pd.DataFrame(table.values())

    adjusted_group_table["elo_rating"] = adjusted_group_table["team"].apply(
        lambda x: get_team_stats(x)["elo_rating"]
    )

    adjusted_group_table["squad_strength_score"] = adjusted_group_table["team"].apply(
        lambda x: get_squad_strength_score(
            x,
            use_injury_adjustment=use_injury_adjustment,
        )
    )

    adjusted_group_table = adjusted_group_table.sort_values(
        [
            "squad_adjusted_expected_points",
            "squad_adjusted_expected_wins",
            "elo_rating",
        ],
        ascending=False,
    ).reset_index(drop=True)

    adjusted_group_table["squad_adjusted_expected_rank"] = (
        adjusted_group_table.index + 1
    )

    display_cols = [
        "squad_adjusted_expected_rank",
        "team",
        "elo_rating",
        "squad_strength_score",
        "squad_adjusted_expected_points",
        "squad_adjusted_expected_wins",
        "squad_adjusted_expected_draws",
        "squad_adjusted_expected_losses",
    ]

    adjusted_group_table = adjusted_group_table[display_cols]

    numeric_cols = [
        "elo_rating",
        "squad_strength_score",
        "squad_adjusted_expected_points",
        "squad_adjusted_expected_wins",
        "squad_adjusted_expected_draws",
        "squad_adjusted_expected_losses",
    ]

    adjusted_group_table[numeric_cols] = adjusted_group_table[numeric_cols].round(3)

    return adjusted_matches, adjusted_group_table


# =========================
# Monte Carlo group simulation
# =========================

def simulate_group_monte_carlo(
    group_name,
    n_simulations=10000,
    adjustment_weight=0.25,
    use_injury_adjustment=False,
    random_seed=42,
):
    """
    Simulate one 4-team group many times using match win/draw/loss probabilities.

    Note:
    This is a W/D/L-based simulation. It does not simulate exact scores,
    so real FIFA tie-breakers such as goal difference and goals scored are approximated.
    Ranking tie-breakers used here:
    points -> wins -> Elo rating -> squad strength -> random tie-breaker.
    """

    rng = np.random.default_rng(random_seed)

    teams = GROUPS_2026[group_name]
    team_to_idx = {team: idx for idx, team in enumerate(teams)}

    fixtures = make_group_fixtures(group_name)

    # Pre-compute fixture probabilities
    fixture_probs = []

    for _, row in fixtures.iterrows():
        team_a = row["team_a"]
        team_b = row["team_b"]

        base_prediction, _ = predict_match_elo(
            team_a=team_a,
            team_b=team_b,
            neutral=True,
            is_world_cup=True,
        )

        adjusted_prediction = adjust_prediction_with_squad_strength(
            base_prediction=base_prediction,
            team_a=team_a,
            team_b=team_b,
            adjustment_weight=adjustment_weight,
            use_injury_adjustment=use_injury_adjustment,
        )

        fixture_probs.append(
            {
                "team_a": team_a,
                "team_b": team_b,
                "idx_a": team_to_idx[team_a],
                "idx_b": team_to_idx[team_b],
                "p_a_win": float(adjusted_prediction.loc[0, "adjusted_probability"]),
                "p_draw": float(adjusted_prediction.loc[1, "adjusted_probability"]),
                "p_b_win": float(adjusted_prediction.loc[2, "adjusted_probability"]),
            }
        )

    n_teams = len(teams)

    first_place_count = np.zeros(n_teams)
    second_place_count = np.zeros(n_teams)
    qualified_count = np.zeros(n_teams)
    eliminated_count = np.zeros(n_teams)

    total_points = np.zeros(n_teams)
    total_rank = np.zeros(n_teams)

    elo_scores = np.array(
        [float(get_team_stats(team)["elo_rating"]) for team in teams]
    )

    squad_scores = np.array(
        [
            float(
                get_squad_strength_score(
                    team,
                    use_injury_adjustment=use_injury_adjustment,
                )
            )
            for team in teams
        ]
    )

    for _ in range(n_simulations):
        points = np.zeros(n_teams)
        wins = np.zeros(n_teams)
        draws = np.zeros(n_teams)
        losses = np.zeros(n_teams)

        for fixture in fixture_probs:
            idx_a = fixture["idx_a"]
            idx_b = fixture["idx_b"]

            outcome = rng.choice(
                [0, 1, 2],
                p=[
                    fixture["p_a_win"],
                    fixture["p_draw"],
                    fixture["p_b_win"],
                ],
            )

            # 0 = Team A win, 1 = draw, 2 = Team B win
            if outcome == 0:
                points[idx_a] += 3
                wins[idx_a] += 1
                losses[idx_b] += 1
            elif outcome == 1:
                points[idx_a] += 1
                points[idx_b] += 1
                draws[idx_a] += 1
                draws[idx_b] += 1
            else:
                points[idx_b] += 3
                wins[idx_b] += 1
                losses[idx_a] += 1

        # Approximate ranking:
        # points -> wins -> Elo -> squad strength -> random tie-breaker
        random_tiebreaker = rng.random(n_teams)

        order = np.lexsort(
            (
                -random_tiebreaker,
                -squad_scores,
                -elo_scores,
                -wins,
                -points,
            )
        )

        ranks = np.empty(n_teams)
        ranks[order] = np.arange(1, n_teams + 1)

        first_place_count[order[0]] += 1
        second_place_count[order[1]] += 1
        qualified_count[order[:2]] += 1
        eliminated_count[order[2:]] += 1

        total_points += points
        total_rank += ranks

    result = pd.DataFrame(
        {
            "team": teams,
            "elo_rating": elo_scores,
            "squad_strength_score": squad_scores,
            "average_points": total_points / n_simulations,
            "average_rank": total_rank / n_simulations,
            "group_winner_%": first_place_count / n_simulations * 100,
            "runner_up_%": second_place_count / n_simulations * 100,
            "qualification_%": qualified_count / n_simulations * 100,
            "elimination_%": eliminated_count / n_simulations * 100,
        }
    )

    result = result.sort_values(
        ["qualification_%", "group_winner_%", "average_points"],
        ascending=False,
    ).reset_index(drop=True)

    result["simulated_rank"] = result.index + 1

    display_cols = [
        "simulated_rank",
        "team",
        "elo_rating",
        "squad_strength_score",
        "average_points",
        "average_rank",
        "group_winner_%",
        "runner_up_%",
        "qualification_%",
        "elimination_%",
    ]

    result = result[display_cols]

    numeric_cols = [
        "elo_rating",
        "squad_strength_score",
        "average_points",
        "average_rank",
        "group_winner_%",
        "runner_up_%",
        "qualification_%",
        "elimination_%",
    ]

    result[numeric_cols] = result[numeric_cols].round(2)

    return result




def get_knockout_advance_probabilities(
    team_a,
    team_b,
    adjustment_weight=0.25,
    use_injury_adjustment=False,
):
    """
    Return adjusted knockout advance probabilities for two teams.

    In this simplified model:
    - regulation win means direct advance
    - draw probability is split 50/50 between the two teams
    - penalties / extra time are not modeled separately
    """

    prediction, _ = predict_match_elo(
        team_a=team_a,
        team_b=team_b,
        neutral=True,
        is_world_cup=True,
    )

    adjusted_prediction = adjust_prediction_with_squad_strength(
        base_prediction=prediction,
        team_a=team_a,
        team_b=team_b,
        adjustment_weight=adjustment_weight,
        use_injury_adjustment=use_injury_adjustment,
    )

    p_a_win = float(adjusted_prediction.loc[0, "adjusted_probability"])
    p_draw = float(adjusted_prediction.loc[1, "adjusted_probability"])
    p_b_win = float(adjusted_prediction.loc[2, "adjusted_probability"])

    p_a_advance = p_a_win + 0.5 * p_draw
    p_b_advance = p_b_win + 0.5 * p_draw

    return {
        "team_a": team_a,
        "team_b": team_b,
        "p_team_a_advance": p_a_advance,
        "p_team_b_advance": p_b_advance,
    }


def make_knockout_prediction(
    team_a,
    team_b,
    adjustment_weight=0.25,
    use_injury_adjustment=False,
):
    prediction, features = predict_match_elo(
        team_a=team_a,
        team_b=team_b,
        neutral=True,
        is_world_cup=True,
    )

    adjusted_prediction = adjust_prediction_with_squad_strength(
        base_prediction=prediction,
        team_a=team_a,
        team_b=team_b,
        adjustment_weight=adjustment_weight,
        use_injury_adjustment=use_injury_adjustment,
    )

    p_a_win = prediction.loc[0, "probability"]
    p_draw = prediction.loc[1, "probability"]
    p_b_win = prediction.loc[2, "probability"]

    p_a_advance = p_a_win + 0.5 * p_draw
    p_b_advance = p_b_win + 0.5 * p_draw

    base_advance = pd.DataFrame(
        {
            "team": [team_a, team_b],
            "advance_probability": [p_a_advance, p_b_advance],
            "advance_probability_percent": [
                round(p_a_advance * 100, 1),
                round(p_b_advance * 100, 1),
            ],
        }
    )

    adj_p_a_win = adjusted_prediction.loc[0, "adjusted_probability"]
    adj_p_draw = adjusted_prediction.loc[1, "adjusted_probability"]
    adj_p_b_win = adjusted_prediction.loc[2, "adjusted_probability"]

    adj_p_a_advance = adj_p_a_win + 0.5 * adj_p_draw
    adj_p_b_advance = adj_p_b_win + 0.5 * adj_p_draw

    adjusted_advance = pd.DataFrame(
        {
            "team": [team_a, team_b],
            "adjusted_advance_probability": [
                adj_p_a_advance,
                adj_p_b_advance,
            ],
            "adjusted_advance_probability_percent": [
                round(adj_p_a_advance * 100, 1),
                round(adj_p_b_advance * 100, 1),
            ],
        }
    )

    return prediction, adjusted_prediction, base_advance, adjusted_advance, features



# =========================
# Simplified full tournament Monte Carlo simulation
# =========================

def simulate_tournament_monte_carlo(
    n_simulations=1000,
    adjustment_weight=0.25,
    use_injury_adjustment=False,
    random_seed=42,
):
    """
    Simplified full-tournament Monte Carlo simulation.

    V1 simplification:
    - Each group has 4 teams.
    - Top 2 teams in each group qualify, giving 24 teams.
    - Top 8 group winners receive a bye to Round of 16.
    - Remaining 16 teams play a Round of 24.
    - Later rounds are simulated by reseeding: highest seed vs lowest seed.

    Important:
    This is not a full official bracket implementation.
    It is a stable first version for estimating tournament-path probabilities.
    """

    rng = np.random.default_rng(random_seed)

    all_teams = []
    team_group = {}

    for group_name, teams in GROUPS_2026.items():
        for team in teams:
            all_teams.append(team)
            team_group[team] = group_name

    all_teams = list(dict.fromkeys(all_teams))
    team_to_idx = {team: idx for idx, team in enumerate(all_teams)}
    n_teams = len(all_teams)

    group_winner_count = np.zeros(n_teams)
    group_runner_up_count = np.zeros(n_teams)
    round_of_24_count = np.zeros(n_teams)
    round_of_16_count = np.zeros(n_teams)
    quarter_final_count = np.zeros(n_teams)
    semi_final_count = np.zeros(n_teams)
    final_count = np.zeros(n_teams)
    champion_count = np.zeros(n_teams)

    total_group_points = np.zeros(n_teams)

    # Static team strength values for tie-break and seeding
    elo_lookup = {
        team: float(get_team_stats(team)["elo_rating"])
        for team in all_teams
    }

    squad_lookup = {
        team: float(
            get_squad_strength_score(
                team,
                use_injury_adjustment=use_injury_adjustment,
            )
        )
        for team in all_teams
    }

    # Pre-compute group fixture probabilities
    group_fixture_probs = {}

    for group_name in GROUPS_2026.keys():
        fixtures = make_group_fixtures(group_name)
        fixture_probs = []

        for _, row in fixtures.iterrows():
            team_a = row["team_a"]
            team_b = row["team_b"]

            base_prediction, _ = predict_match_elo(
                team_a=team_a,
                team_b=team_b,
                neutral=True,
                is_world_cup=True,
            )

            adjusted_prediction = adjust_prediction_with_squad_strength(
                base_prediction=base_prediction,
                team_a=team_a,
                team_b=team_b,
                adjustment_weight=adjustment_weight,
                use_injury_adjustment=use_injury_adjustment,
            )

            fixture_probs.append(
                {
                    "team_a": team_a,
                    "team_b": team_b,
                    "p_a_win": float(adjusted_prediction.loc[0, "adjusted_probability"]),
                    "p_draw": float(adjusted_prediction.loc[1, "adjusted_probability"]),
                    "p_b_win": float(adjusted_prediction.loc[2, "adjusted_probability"]),
                }
            )

        group_fixture_probs[group_name] = fixture_probs

    # Cache knockout advance probabilities
    advance_cache = {}

    def play_knockout_match(team_a, team_b):
        key = tuple(sorted([team_a, team_b]))

        if key not in advance_cache:
            probs = get_knockout_advance_probabilities(
                key[0],
                key[1],
                adjustment_weight=adjustment_weight,
                use_injury_adjustment=use_injury_adjustment,
            )
            advance_cache[key] = {
                key[0]: probs["p_team_a_advance"],
                key[1]: probs["p_team_b_advance"],
            }

        p_team_a_advance = advance_cache[key][team_a]

        if rng.random() < p_team_a_advance:
            return team_a
        else:
            return team_b

    def sort_by_seed(team_list, seed_rank):
        return sorted(team_list, key=lambda team: seed_rank[team])

    def play_seeded_round(team_list, seed_rank):
        ordered_teams = sort_by_seed(team_list, seed_rank)
        winners = []

        n = len(ordered_teams)

        for i in range(n // 2):
            team_a = ordered_teams[i]
            team_b = ordered_teams[n - 1 - i]
            winners.append(play_knockout_match(team_a, team_b))

        return winners

    for _ in range(n_simulations):
        qualified_rows = []

        # ----- Group stage -----
        for group_name, teams in GROUPS_2026.items():
            points = {team: 0.0 for team in teams}
            wins = {team: 0.0 for team in teams}
            draws = {team: 0.0 for team in teams}
            losses = {team: 0.0 for team in teams}

            for fixture in group_fixture_probs[group_name]:
                team_a = fixture["team_a"]
                team_b = fixture["team_b"]

                outcome = rng.choice(
                    [0, 1, 2],
                    p=[
                        fixture["p_a_win"],
                        fixture["p_draw"],
                        fixture["p_b_win"],
                    ],
                )

                # 0 = Team A win, 1 = draw, 2 = Team B win
                if outcome == 0:
                    points[team_a] += 3
                    wins[team_a] += 1
                    losses[team_b] += 1
                elif outcome == 1:
                    points[team_a] += 1
                    points[team_b] += 1
                    draws[team_a] += 1
                    draws[team_b] += 1
                else:
                    points[team_b] += 3
                    wins[team_b] += 1
                    losses[team_a] += 1

            group_rank_rows = []

            for team in teams:
                idx = team_to_idx[team]
                total_group_points[idx] += points[team]

                group_rank_rows.append(
                    {
                        "team": team,
                        "group": group_name,
                        "points": points[team],
                        "wins": wins[team],
                        "elo_rating": elo_lookup[team],
                        "squad_strength_score": squad_lookup[team],
                        "random_tiebreaker": rng.random(),
                    }
                )

            group_rank_df = pd.DataFrame(group_rank_rows)

            group_rank_df = group_rank_df.sort_values(
                [
                    "points",
                    "wins",
                    "elo_rating",
                    "squad_strength_score",
                    "random_tiebreaker",
                ],
                ascending=False,
            ).reset_index(drop=True)

            group_winner = group_rank_df.loc[0, "team"]
            group_runner_up = group_rank_df.loc[1, "team"]

            group_winner_count[team_to_idx[group_winner]] += 1
            group_runner_up_count[team_to_idx[group_runner_up]] += 1

            for position, row_idx in [("winner", 0), ("runner_up", 1)]:
                row = group_rank_df.loc[row_idx].copy()
                row["group_position"] = position
                row["group_position_rank"] = 1 if position == "winner" else 2
                qualified_rows.append(row.to_dict())

        qualified_df = pd.DataFrame(qualified_rows)

        # Every qualified team has reached the 24-team knockout field
        for team in qualified_df["team"]:
            round_of_24_count[team_to_idx[team]] += 1

        # ----- Seeding -----
        qualified_df["seed_random"] = rng.random(len(qualified_df))

        qualified_df = qualified_df.sort_values(
            [
                "group_position_rank",
                "points",
                "wins",
                "elo_rating",
                "squad_strength_score",
                "seed_random",
            ],
            ascending=[True, False, False, False, False, False],
        ).reset_index(drop=True)

        seed_rank = {
            row["team"]: rank
            for rank, row in qualified_df.reset_index().iterrows()
        }

        group_winners_df = qualified_df[
            qualified_df["group_position"] == "winner"
        ].copy()

        group_winners_df = group_winners_df.sort_values(
            [
                "points",
                "wins",
                "elo_rating",
                "squad_strength_score",
                "seed_random",
            ],
            ascending=[False, False, False, False, False],
        ).reset_index(drop=True)

        bye_teams = group_winners_df.head(8)["team"].tolist()

        round_of_24_teams = [
            team for team in qualified_df["team"].tolist()
            if team not in bye_teams
        ]

        # ----- Round of 24 -----
        round_of_24_winners = play_seeded_round(
            round_of_24_teams,
            seed_rank,
        )

        # ----- Round of 16 -----
        round_of_16_teams = bye_teams + round_of_24_winners

        for team in round_of_16_teams:
            round_of_16_count[team_to_idx[team]] += 1

        quarter_final_teams = play_seeded_round(
            round_of_16_teams,
            seed_rank,
        )

        for team in quarter_final_teams:
            quarter_final_count[team_to_idx[team]] += 1

        # ----- Quarter-finals -----
        semi_final_teams = play_seeded_round(
            quarter_final_teams,
            seed_rank,
        )

        for team in semi_final_teams:
            semi_final_count[team_to_idx[team]] += 1

        # ----- Semi-finals -----
        final_teams = play_seeded_round(
            semi_final_teams,
            seed_rank,
        )

        for team in final_teams:
            final_count[team_to_idx[team]] += 1

        # ----- Final -----
        champion = play_seeded_round(
            final_teams,
            seed_rank,
        )[0]

        champion_count[team_to_idx[champion]] += 1

    result = pd.DataFrame(
        {
            "team": all_teams,
            "group": [team_group[team] for team in all_teams],
            "elo_rating": [elo_lookup[team] for team in all_teams],
            "squad_strength_score": [squad_lookup[team] for team in all_teams],
            "average_group_points": total_group_points / n_simulations,
            "group_winner_%": group_winner_count / n_simulations * 100,
            "group_runner_up_%": group_runner_up_count / n_simulations * 100,
            "round_of_24_%": round_of_24_count / n_simulations * 100,
            "round_of_16_%": round_of_16_count / n_simulations * 100,
            "quarter_final_%": quarter_final_count / n_simulations * 100,
            "semi_final_%": semi_final_count / n_simulations * 100,
            "final_%": final_count / n_simulations * 100,
            "champion_%": champion_count / n_simulations * 100,
        }
    )

    result = result.sort_values(
        ["champion_%", "final_%", "semi_final_%", "quarter_final_%"],
        ascending=False,
    ).reset_index(drop=True)

    result["tournament_rank"] = result.index + 1

    display_cols = [
        "tournament_rank",
        "team",
        "group",
        "elo_rating",
        "squad_strength_score",
        "average_group_points",
        "group_winner_%",
        "group_runner_up_%",
        "round_of_24_%",
        "round_of_16_%",
        "quarter_final_%",
        "semi_final_%",
        "final_%",
        "champion_%",
    ]

    result = result[display_cols]

    numeric_cols = [
        "elo_rating",
        "squad_strength_score",
        "average_group_points",
        "group_winner_%",
        "group_runner_up_%",
        "round_of_24_%",
        "round_of_16_%",
        "quarter_final_%",
        "semi_final_%",
        "final_%",
        "champion_%",
    ]

    result[numeric_cols] = result[numeric_cols].round(2)

    return result


# =========================
# App layout
# =========================

st.title("⚽ 2026 世界杯预测")
st.caption("基于 Elo 评级 + 近 10 场状态的预测模型")

tab_single, tab_group, tab_knockout, tab_tournament, tab_model = st.tabs(
    [
        "单场比赛预测",
        "小组赛预测",
        "淘汰赛预测",
        "整届模拟",
        "模型说明与下一步",
    ]
)


# =========================
# Tab 1: Single match
# =========================

with tab_single:
    st.subheader("单场比赛预测")

    col1, col2, col3 = st.columns(3)

    with col1:
        team_a = st.selectbox(
            "主队",
            TEAM_OPTIONS,
            index=TEAM_OPTIONS.index("Argentina") if "Argentina" in TEAM_OPTIONS else 0,
        )

    with col2:
        team_b = st.selectbox(
            "客队",
            TEAM_OPTIONS,
            index=TEAM_OPTIONS.index("France") if "France" in TEAM_OPTIONS else 1,
        )

    with col3:
        neutral = st.checkbox("中立场地", value=True)
        is_world_cup = st.checkbox("世界杯比赛", value=True)

    if team_a == team_b:
        st.warning("请选择两支不同球队。")
    else:
        prediction, features = predict_match_elo(
            team_a=team_a,
            team_b=team_b,
            neutral=neutral,
            is_world_cup=is_world_cup,
        )

        st.plotly_chart(
            make_probability_chart(prediction),
            use_container_width=True,
        key="single_match_base_probability_chart",
        )

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        metric_col1.metric(
            prediction.loc[0, "result"],
            f"{prediction.loc[0, 'probability_percent']:.1f}%",
        )

        metric_col2.metric(
            prediction.loc[1, "result"],
            f"{prediction.loc[1, 'probability_percent']:.1f}%",
        )

        metric_col3.metric(
            prediction.loc[2, "result"],
            f"{prediction.loc[2, 'probability_percent']:.1f}%",
        )

        st.markdown("### 两队 Elo 与近 10 场状态对比")
        st.dataframe(make_team_comparison(team_a, team_b))

        with st.expander("🔍 查看模型特征与原始概率"):
            st.markdown("#### 本场比赛输入模型的特征")
            st.dataframe(features.round(3))
            st.markdown("#### 原始概率表")
            st.dataframe(prediction)

        st.markdown("### 综合强度 / 阵容修正模块")

        adjustment_weight = st.slider(
            "综合强度修正系数",
            min_value=0.00,
            max_value=0.35,
            value=0.25,
            step=0.05,
            help="数值越大，综合强度对胜负概率的影响越明显。0 表示不修正。",
        )

        use_injury_adjustment_single = st.checkbox(
            "启用 injury adjustment",
            value=False,
            help=(
                "开启后使用 injury_adjusted_squad_strength_score；"
                "目前已人工核查全部 48 队；"
                "伤病状态会随赛前新闻变化，建议在比赛前定期复核。"
            ),
        )

        if use_injury_adjustment_single:
            st.caption(
                "当前为 injury adjustment 模式：综合强度会扣除 injury_impact_score "
                "和 star unavailable penalty。"
            )

        adjusted_prediction = adjust_prediction_with_squad_strength(
            base_prediction=prediction,
            team_a=team_a,
            team_b=team_b,
            adjustment_weight=adjustment_weight,
            use_injury_adjustment=use_injury_adjustment_single,
        )

        adjusted_display = adjusted_prediction[
            [
                "result",
                "adjusted_probability",
                "adjusted_probability_percent",
            ]
        ].rename(
            columns={
                "adjusted_probability": "probability",
                "adjusted_probability_percent": "probability_percent",
            }
        )

        st.markdown("#### 综合强度修正后概率图")
        st.plotly_chart(
            make_probability_chart(adjusted_display),
            use_container_width=True,
        key="single_match_adjusted_probability_chart",
        )

        with st.expander("📊 查看综合强度详细对比"):
            comparison_probability = adjusted_prediction[
                [
                    "result",
                    "probability_percent",
                    "adjusted_probability_percent",
                    "squad_strength_team_a",
                    "squad_strength_team_b",
                    "squad_strength_diff",
                ]
            ].copy()

            comparison_probability = comparison_probability.rename(
                columns={
                    "probability_percent": "base_elo_probability_%",
                    "adjusted_probability_percent": "squad_adjusted_probability_%",
                }
            )

            st.markdown("#### 基础 Elo 概率 vs 综合强度修正后概率")
            st.dataframe(comparison_probability)

            st.markdown("#### 两队综合强度 / 阵容字段对比")
            st.dataframe(make_squad_strength_comparison(team_a, team_b))

            st.caption(
                "数据来源：FIFA official API（FIFA rank / FIFA points）；Transfermarkt（squad value）；proxy rating 由 FIFA points 和 Transfermarkt market value 推算；伤病数据已根据 48 队 injury audit 更新，建议赛前定期复核。"
            )

# =========================
# Tab 2: Group stage
# =========================

with tab_group:
    st.subheader("小组赛预测")

    selected_group = st.selectbox(
        "选择小组",
        list(GROUPS_2026.keys()),
    )

    group_adjustment_weight = st.slider(
        "小组赛综合强度修正系数",
        min_value=0.00,
        max_value=0.35,
        value=0.25,
        step=0.05,
        help="数值越大，综合强度对小组赛概率和预期积分的影响越明显。0 表示不修正。",
    )

    use_injury_adjustment_group = st.checkbox(
        "小组赛启用 injury adjustment",
        value=False,
        help=(
            "开启后小组赛综合强度使用 injury_adjusted_squad_strength_score；"
            "目前已人工核查全部 48 队；"
            "伤病状态会随赛前新闻变化，建议在比赛前定期复核。"
        ),
    )

    if use_injury_adjustment_group:
        st.caption(
            "当前小组赛为 injury adjustment 模式：综合强度会扣除 injury_impact_score "
            "和 star unavailable penalty。"
        )

    st.markdown(f"### {selected_group} 参赛球队：{', '.join(GROUPS_2026[selected_group])}")

    group_predictions = predict_group_matches(selected_group)
    expected_table = make_expected_group_table(selected_group)

    (
        squad_adjusted_matches,
        squad_adjusted_group_table,
    ) = make_squad_adjusted_group_table(
        selected_group,
        adjustment_weight=group_adjustment_weight,
        use_injury_adjustment=use_injury_adjustment_group,
    )

    st.markdown("### 基础 Elo 小组赛 6 场比赛胜 / 平 / 负概率")

    st.dataframe(
        group_predictions[
            [
                "match_no",
                "team_a",
                "team_b",
                "team_a_win_%",
                "draw_%",
                "team_b_win_%",
            ]
        ]
    )

    st.markdown("### 基础 Elo 小组预期积分排名")

    st.dataframe(expected_table)

    fig_group = px.bar(
        expected_table,
        x="team",
        y="expected_points",
        text="expected_points",
        title=f"{selected_group} 基础 Elo 预期积分",
    )

    fig_group.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
    )

    fig_group.update_layout(
        yaxis_title="预期积分",
        xaxis_title="球队",
    )

    st.plotly_chart(fig_group, use_container_width=True, key="group_base_expected_points_chart")

    st.markdown("### 综合强度修正后小组赛 6 场比赛胜 / 平 / 负概率")

    st.dataframe(
        squad_adjusted_matches[
            [
                "match_no",
                "team_a",
                "team_b",
                "team_a_win_%",
                "draw_%",
                "team_b_win_%",
            ]
        ]
    )

    st.markdown("### 综合强度修正后小组预期积分排名")

    st.dataframe(squad_adjusted_group_table)

    fig_group_adjusted = px.bar(
        squad_adjusted_group_table,
        x="team",
        y="squad_adjusted_expected_points",
        text="squad_adjusted_expected_points",
        title=f"{selected_group} 综合强度修正后预期积分",
    )

    fig_group_adjusted.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
    )

    fig_group_adjusted.update_layout(
        yaxis_title="综合强度修正后预期积分",
        xaxis_title="球队",
    )

    st.plotly_chart(fig_group_adjusted, use_container_width=True, key="group_adjusted_expected_points_chart")

    with st.expander("📋 查看 Elo vs 综合强度排名对比"):
        rank_comparison = expected_table[
            [
                "expected_rank",
                "team",
                "expected_points",
            ]
        ].merge(
            squad_adjusted_group_table[
                [
                    "squad_adjusted_expected_rank",
                    "team",
                    "squad_adjusted_expected_points",
                    "squad_strength_score",
                ]
            ],
            on="team",
            how="left",
        )

        rank_comparison["rank_change_after_squad_adjustment"] = (
            rank_comparison["expected_rank"]
            - rank_comparison["squad_adjusted_expected_rank"]
        )

        rank_comparison = rank_comparison.sort_values(
            "squad_adjusted_expected_rank"
        ).reset_index(drop=True)

        st.dataframe(rank_comparison)

        st.caption(
            "rank_change > 0 表示修正后排名上升；< 0 表示下降。综合强度由 FIFA 积分、Transfermarkt 市值和 proxy rating 共同生成；伤病数据已根据 48 队 audit 更新，建议赛前定期复核。"
        )


    st.markdown("### Monte Carlo 小组出线概率模拟")

    monte_carlo_simulations = st.slider(
        "Monte Carlo 模拟次数",
        min_value=1000,
        max_value=30000,
        value=10000,
        step=1000,
        help=(
            "模拟次数越多，结果越稳定，但运行时间也会更长。"
            "当前模拟基于胜 / 平 / 负概率，不模拟具体比分，因此净胜球等真实 tie-breakers 是近似处理。"
        ),
    )

    run_group_monte_carlo = st.button(
        "运行 Monte Carlo 模拟",
        key="run_group_monte_carlo",
    )

    if run_group_monte_carlo:
        with st.spinner("正在模拟小组赛，请稍等..."):
            monte_carlo_group_table = simulate_group_monte_carlo(
                selected_group,
                n_simulations=int(monte_carlo_simulations),
                adjustment_weight=group_adjustment_weight,
                use_injury_adjustment=use_injury_adjustment_group,
                random_seed=42,
            )

        st.dataframe(monte_carlo_group_table)

        fig_group_monte_carlo = px.bar(
            monte_carlo_group_table,
            x="team",
            y="qualification_%",
            text="qualification_%",
            title=f"{selected_group} Monte Carlo 出线概率",
        )

        fig_group_monte_carlo.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
        )

        fig_group_monte_carlo.update_layout(
            yaxis_title="出线概率 (%)",
            xaxis_title="球队",
        )

        st.plotly_chart(
            fig_group_monte_carlo,
            use_container_width=True,
            key="group_monte_carlo_qualification_chart",
        )

        st.caption(
            "Monte Carlo 结果表示在当前模型概率下反复模拟小组赛后的出线概率。"
            "由于当前版本只模拟胜 / 平 / 负，不模拟具体比分，因此同分时的排名规则使用 points、wins、Elo、综合强度和随机 tie-breaker 近似处理。"
        )


# =========================
# Tab 3: Knockout stage
# =========================

with tab_knockout:
    st.subheader("淘汰赛预测")

    col1, col2 = st.columns(2)

    with col1:
        ko_team_a = st.selectbox(
            "淘汰赛 主队",
            TEAM_OPTIONS,
            index=TEAM_OPTIONS.index("Canada") if "Canada" in TEAM_OPTIONS else 0,
        )

    with col2:
        ko_team_b = st.selectbox(
            "淘汰赛 客队",
            TEAM_OPTIONS,
            index=TEAM_OPTIONS.index("Senegal") if "Senegal" in TEAM_OPTIONS else 1,
        )

    if ko_team_a == ko_team_b:
        st.warning("请选择两支不同球队。")
    else:
        ko_adjustment_weight = st.slider(
            "淘汰赛综合强度修正系数",
            min_value=0.00,
            max_value=0.35,
            value=0.25,
            step=0.05,
            help="数值越大，综合强度对淘汰赛晋级概率的影响越明显。0 表示不修正。",
        )

        use_injury_adjustment_knockout = st.checkbox(
            "淘汰赛启用 injury adjustment",
            value=False,
            help=(
                "开启后淘汰赛综合强度使用 injury_adjusted_squad_strength_score；"
                "目前已人工核查全部 48 队；"
                "伤病状态会随赛前新闻变化，建议在比赛前定期复核。"
            ),
        )

        if use_injury_adjustment_knockout:
            st.caption(
                "当前淘汰赛为 injury adjustment 模式：综合强度会扣除 injury_impact_score "
                "和 star unavailable penalty。"
            )

        (
            ko_prediction,
            ko_adjusted_prediction,
            ko_advance,
            ko_adjusted_advance,
            ko_features,
        ) = make_knockout_prediction(
            ko_team_a,
            ko_team_b,
            adjustment_weight=ko_adjustment_weight,
            use_injury_adjustment=use_injury_adjustment_knockout,
        )

        st.markdown("### 90 分钟基础 Elo 胜 / 平 / 负概率")

        st.plotly_chart(
            make_probability_chart(ko_prediction),
            use_container_width=True,
        key="knockout_base_probability_chart",
        )

        st.markdown("### 90 分钟综合强度修正后胜 / 平 / 负概率")

        ko_adjusted_display = ko_adjusted_prediction[
            [
                "result",
                "adjusted_probability",
                "adjusted_probability_percent",
            ]
        ].rename(
            columns={
                "adjusted_probability": "probability",
                "adjusted_probability_percent": "probability_percent",
            }
        )

        st.plotly_chart(
            make_probability_chart(ko_adjusted_display),
            use_container_width=True,
        key="knockout_adjusted_probability_chart",
        )

        with st.expander("🔍 查看基础 vs 修正概率对比"):
            st.dataframe(
                ko_adjusted_prediction[
                    [
                        "result",
                        "probability_percent",
                        "adjusted_probability_percent",
                        "squad_strength_team_a",
                        "squad_strength_team_b",
                        "squad_strength_diff",
                    ]
                ].rename(
                    columns={
                        "probability_percent": "base_elo_probability_%",
                        "adjusted_probability_percent": "squad_adjusted_probability_%",
                    }
                )
            )

        st.markdown("### 基础近似晋级概率")

        st.caption(
            "基础晋级概率采用简化规则：球队晋级概率 = 90分钟获胜概率 + 0.5 × 平局概率。"
        )

        fig_advance = px.bar(
            ko_advance,
            x="team",
            y="advance_probability_percent",
            text="advance_probability_percent",
            title="基础近似晋级概率",
        )

        fig_advance.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
        )

        fig_advance.update_layout(
            yaxis_title="晋级概率 (%)",
            xaxis_title="球队",
            yaxis_range=[0, 100],
        )

        st.plotly_chart(fig_advance, use_container_width=True, key="knockout_base_advance_chart")

        st.markdown("### 综合强度修正后近似晋级概率")

        fig_adjusted_advance = px.bar(
            ko_adjusted_advance,
            x="team",
            y="adjusted_advance_probability_percent",
            text="adjusted_advance_probability_percent",
            title="综合强度修正后近似晋级概率",
        )

        fig_adjusted_advance.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
        )

        fig_adjusted_advance.update_layout(
            yaxis_title="修正后晋级概率 (%)",
            xaxis_title="球队",
            yaxis_range=[0, 100],
        )

        st.plotly_chart(fig_adjusted_advance, use_container_width=True, key="knockout_adjusted_advance_chart")

        with st.expander("📊 查看详细数据"):
            st.markdown("#### 修正后晋级概率表")
            st.dataframe(ko_adjusted_advance)
            st.markdown("#### 本场比赛输入模型的特征")
            st.dataframe(ko_features.round(3))
            st.markdown("#### 两队综合强度 / 阵容字段对比")
            st.dataframe(make_squad_strength_comparison(ko_team_a, ko_team_b))


# =========================
# Tab 4: Full tournament Monte Carlo
# =========================

with tab_tournament:
    st.subheader("整届世界杯 Monte Carlo 模拟")

    st.caption("每组前 2 名出线共 24 队；积分最佳的 8 个小组第一获轮空直接晋级 16 强；其余 16 队先打附加赛。概率为估算值，非官方 bracket。")
    with st.expander("🔧 查看模拟方法详情"):
        st.markdown(
            """
- 每组前 2 名出线，共 24 队进入淘汰阶段
- 积分最佳的 8 个小组第一直接进入 Round of 16（轮空）
- 其余 16 队打 Round of 24，胜者与轮空队共同组成 16 强
- 后续淘汰赛重新种子排序：最高种子 vs 最低种子
- V1 简化版：不模拟具体比分，tie-breakers 用积分、胜场、Elo、综合强度近似处理
            """
        )

    tournament_col1, tournament_col2, tournament_col3 = st.columns(3)

    with tournament_col1:
        tournament_simulations = st.slider(
            "整届模拟次数",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            key="tournament_monte_carlo_simulations",
            help="模拟次数越多结果越稳定，但运行时间也会更长。",
        )

    with tournament_col2:
        tournament_adjustment_weight = st.slider(
            "整届模拟综合强度修正系数",
            min_value=0.00,
            max_value=0.35,
            value=0.25,
            step=0.05,
            key="tournament_adjustment_weight",
            help="控制综合强度对整届模拟中比赛概率的影响。",
        )

    with tournament_col3:
        use_injury_adjustment_tournament = st.checkbox(
            "整届模拟启用 injury adjustment",
            value=False,
            key="tournament_injury_adjustment",
            help=(
                "开启后整届模拟使用 injury_adjusted_squad_strength_score；"
                "目前已人工核查全部 48 队；"
                "伤病状态会随赛前新闻变化，建议在比赛前定期复核。"
            ),
        )

    run_tournament_simulation = st.button(
        "运行整届 Monte Carlo 模拟",
        key="run_tournament_monte_carlo",
    )

    if run_tournament_simulation:
        with st.spinner("正在模拟整届世界杯，请稍等..."):
            tournament_table = simulate_tournament_monte_carlo(
                n_simulations=int(tournament_simulations),
                adjustment_weight=tournament_adjustment_weight,
                use_injury_adjustment=use_injury_adjustment_tournament,
                random_seed=42,
            )

        st.markdown("### 冠军概率 Top 20")

        champion_top20 = tournament_table.head(20)

        st.dataframe(champion_top20)

        fig_tournament_champion = px.bar(
            champion_top20,
            x="team",
            y="champion_%",
            text="champion_%",
            title="Monte Carlo 冠军概率 — Top 20",
        )

        fig_tournament_champion.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
        )

        fig_tournament_champion.update_layout(
            yaxis_title="冠军概率 (%)",
            xaxis_title="球队",
        )

        st.plotly_chart(
            fig_tournament_champion,
            use_container_width=True,
            key="tournament_monte_carlo_champion_chart",
        )

        st.markdown("### 全部 48 队整届模拟结果")

        st.dataframe(tournament_table)

        st.caption(
            "解释：round_of_24_% 表示进入 24 队淘汰阶段的概率；"
            "round_of_16_% 表示进入 16 强的概率；champion_% 表示夺冠概率。"
            "当前 V1 不模拟具体比分和真实官方完整 bracket，后续可以继续升级。"
        )


# =========================
# Tab 4: Model explanation
# =========================

with tab_model:
    st.subheader("模型说明")

    st.markdown(
        """
### 当前模型

**Elo + 近 10 场状态模型**，预测三分类结果（胜 / 平 / 负）。

模型基于两队特征差值进行预测，核心特征为 Elo 差值与近期状态统计差值。

---

### 模型表现

| 模型 | Accuracy | Log Loss |
|------|----------|----------|
| 仅近 10 场状态 | 0.5435 | 0.9687 |
| Elo + 近 10 场状态 | 0.6008 | 0.8837 |

加入 Elo 后预测准确率显著提升。

---

### 关于平局预测

平局概率通常在 20%–28% 之间，很少成为三类中最高概率，因此建议参考概率分布而非单一预测结果。

---

### 综合强度修正

在 Elo 基础预测之上叠加综合强度修正模块（权重可调），使用 FIFA 积分、Transfermarkt 市值和 proxy rating 共同生成的 strength score 对胜负概率进行微调。
        """
    )

    with st.expander("📐 查看模型特征列表"):
        st.markdown(
            """
模型输入特征（均为 A 队 − B 队差值形式）：

- `elo_diff`
- `goals_for_roll10_diff`
- `goals_against_roll10_diff`
- `points_roll10_diff`
- `win_rate_roll10_diff`
- `draw_rate_roll10_diff`
- `loss_rate_roll10_diff`
- `neutral`（中立场地标志）
- `is_world_cup`（世界杯比赛标志）
            """
        )