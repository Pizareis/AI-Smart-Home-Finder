from src.nlp_parser import parse_budget, parse_districts, parse_priorities, parse_room_count, parse_user_input


def test_parse_budget_plain_tl():
    assert parse_budget("Aylık maksimum 20.000 TL verebilirim.") == 20000.0


def test_parse_budget_bin_tl():
    assert parse_budget("Bütçem 15 bin TL.") == 15000.0


def test_parse_budget_none():
    assert parse_budget("Bütçe belirtmedim.") is None


def test_parse_room_count_multiple():
    assert parse_room_count("1+1 veya 2+1 istiyorum.") == ["1+1", "2+1"]


def test_parse_room_count_studio():
    assert parse_room_count("Stüdyo arıyorum.") == ["studio"]


def test_parse_districts_target_and_alternatives():
    target, alternatives = parse_districts(
        "Özlüce'de oturmak istiyorum ama Görükle ya da Beşevler'i de değerlendirebilirim."
    )
    assert target == "Özlüce"
    assert alternatives == ["Görükle", "Beşevler"]


def test_parse_districts_none_found():
    target, alternatives = parse_districts("Bilinmeyen bir yerde oturuyorum.")
    assert target is None
    assert alternatives == []


def test_parse_priorities():
    assert parse_priorities("Ulaşım ve güvenlik benim için önemli.") == ["transport", "safety"]


def test_parse_user_input_full_profile():
    text = (
        "24 yaşındayım, Bursa'da öğrenciyim. Üniversitem Nilüfer'de. "
        "Aylık maksimum 20.000 TL verebilirim. 1+1 veya 2+1 istiyorum. "
        "Ulaşım ve güvenlik benim için önemli."
    )
    profile = parse_user_input(text)
    assert profile.budget_max == 20000.0
    assert profile.room_options == ["1+1", "2+1"]
    assert profile.target_district == "Nilüfer"
    assert profile.priorities == ["transport", "safety"]
