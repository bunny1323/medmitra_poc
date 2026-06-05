import pytest
from app.services.query_classifier import QueryClassifier
from app.core.exceptions import MedicalSafetyException
from app.models.enums import AgeGroup, TopicGroup

@pytest.fixture
def classifier():
    return QueryClassifier()

def test_classify_child_routing(classifier):
    res = classifier.classify("cough", AgeGroup.CHILD, age_years=4)
    assert res["age_group"] == AgeGroup.CHILD
    assert res["topic_group"] == TopicGroup.PEDIATRIC
    assert res["category"] == "child"

def test_classify_adult_routing(classifier):
    res = classifier.classify("cough", AgeGroup.ADULT, age_years=30)
    assert res["age_group"] == AgeGroup.ADULT
    assert res["topic_group"] is None
    assert res["category"] == "adult"

def test_infant_safety_throws_error(classifier):
    with pytest.raises(MedicalSafetyException) as exc:
        classifier.classify("cough", AgeGroup.CHILD, age_months=1)
    assert exc.value.code == "INFANT_SAFETY_LIMITATION"
