from aml_graphrag.domain.judging import position_bias, score_pair, win_rates


def test_a_judge_that_always_picks_the_first_answer_scores_a_tie():
    first = {"comprehensiveness": 1}
    swapped = {"comprehensiveness": 1}  # B was shown first and won
    assert score_pair(first, swapped) == {"comprehensiveness": 0.5}


def test_consistent_wins_and_ties():
    assert score_pair({"c": 1}, {"c": 2}) == {"c": 1.0}
    assert score_pair({"c": 0}, {"c": 0}) == {"c": 0.5}
    assert score_pair({"c": 2}, {"c": 1}) == {"c": 0.0}


def test_win_rates_and_position_bias():
    assert win_rates([{"c": 1.0}, {"c": 0.5}]) == {"c": 0.75}
    assert position_bias([{"c": 1}, {"c": 1}], [{"c": 1}, {"c": 2}]) == 0.5
