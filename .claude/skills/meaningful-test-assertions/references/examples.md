# Before and After Examples

This document shows concrete examples of improving test assertions from weak to meaningful.

## Example 1: Existence Check → Specific Value

### Before
```python
def test_search_entities():
    results = graph.search("machine learning")
    assert results is not None
```

### After (with captured value: results = [Entity(name="Machine Learning")])
```python
def test_search_entities():
    results = graph.search("machine learning")
    assert len(results) == 1
    assert results[0].name == "Machine Learning"
```

---

## Example 2: Type Check → Property Check

### Before
```python
def test_create_user():
    user = create_user("alice@example.com")
    assert isinstance(user, User)
```

### After (with captured value: user = User(id="user-1", email="alice@example.com"))
```python
def test_create_user():
    user = create_user("alice@example.com")
    assert user.email == "alice@example.com"
    assert user.id.startswith("user-")
```

---

## Example 3: Lenient Length → Exact Count

### Before
```python
def test_get_all_users():
    users = get_all_users()
    assert len(users) >= 1
```

### After (with captured value: len(users) = 5)
```python
def test_get_all_users():
    users = get_all_users()
    assert len(users) == 5
```

---

## Example 4: Substring Check → Exact Match

### Before
```python
def test_api_response():
    response = api_call()
    assert "success" in response
```

### After (with captured value: response = {"status": "success", "data": [...]})
```python
def test_api_response():
    response = api_call()
    assert response["status"] == "success"
    assert "data" in response
```

---

## Example 5: Chained Weak Assertions → Split Concrete Assertions

### Before
```python
def test_process_results():
    results = processor.process()
    assert results is not None and len(results) > 0
```

### After (with captured values: results = ["a", "b", "c"])
```python
def test_process_results():
    results = processor.process()
    assert len(results) == 3
    assert results == ["a", "b", "c"]
```

---

## Example 6: Magic Numbers → Named Constants

### Before
```python
def test_pagination():
    page = get_page(1)
    assert len(page.items) == 10
```

### After
```python
PAGE_SIZE = 10

def test_pagination():
    page = get_page(1)
    assert len(page.items) == PAGE_SIZE
```

---

## Example 7: Complex Weak Assertion → Multiple Specific Assertions

### Before
```python
def test_graph_query():
    result = graph.query("machine learning")
    assert result is not None
    assert isinstance(result, list)
    assert len(result) >= 1
```

### After (with captured values: result = [Entity(id=1, name="Machine Learning"), Entity(id=2, name="Python")])
```python
def test_graph_query():
    result = graph.query("machine learning")
    assert len(result) == 2
    assert result[0].name == "Machine Learning"
    assert result[1].name == "Python"
```

---

## Example 8: Negation Check → Positive Assertion

### Before
```python
def test_validate_email():
    is_valid = validate_email("test@example.com")
    assert is_valid is not None
    assert "error" not in is_valid.message
```

### After (with captured values: is_valid = ValidationResult(valid=True, message="Email is valid"))
```python
def test_validate_email():
    is_valid = validate_email("test@example.com")
    assert is_valid.valid == True
    assert is_valid.message == "Email is valid"
```

---

## Example 9: Truthiness Check → Concrete Assertion

### Before
```python
def test_filter_results():
    filtered = filter_results(results, active=True)
    assert filtered
```

### After (with captured values: filtered = [result for result in results if result.active])
```python
def test_filter_results():
    filtered = filter_results(results, active=True)
    assert len(filtered) == 3
    assert all(r.active for r in filtered)
```

---

## Example 10: Mixed Issues → Complete Refactor

### Before
```python
def test_search_with_filters():
    results = search(topic="ML", active=True)
    assert results is not None
    assert len(results) >= 1
    assert results[0].topic == "ML"
```

### After (with captured values: results = [Entity(id=1, topic="Machine Learning", active=True), Entity(id=2, topic="Python ML", active=True)])
```python
SEARCH_RESULTS_COUNT = 2

def test_search_with_filters():
    results = search(topic="ML", active=True)
    assert len(results) == SEARCH_RESULTS_COUNT
    assert results[0].topic == "Machine Learning"
    assert results[0].active == True
    assert results[1].topic == "Python ML"
```

---

## Complex Example: Multiple Variables in One Assertion

### Before
```python
def test_user_permissions():
    user = get_user(123)
    assert user is not None and user.admin is True and len(user.permissions) >= 5
```

### After (with captured values: user=User(id=123, name="admin", admin=True, permissions=["read", "write", "delete", "manage", "audit"]))
```python
ADMIN_PERMISSION_COUNT = 5

def test_user_permissions():
    user = get_user(123)
    assert user.id == 123
    assert user.name == "admin"
    assert user.admin == True
    assert len(user.permissions) == ADMIN_PERMISSION_COUNT
    assert "read" in user.permissions
    assert "write" in user.permissions
```

---

## Example: Testing with Mocked Dependencies

### Before
```python
@patch('external_api.get_data')
def test_integration(mock_get):
    mock_get.return_value = {"data": [1, 2, 3, 4, 5, 6, 7]}
    result = process_api_data()
    assert result is not None
```

### After (with captured value: result = {"processed": True, "count": 7})
```python
API_DATA_COUNT = 7

@patch('external_api.get_data')
def test_integration(mock_get):
    mock_get.return_value = {"data": [1, 2, 3, 4, 5, 6, 7]}
    result = process_api_data()
    assert result["processed"] == True
    assert result["count"] == API_DATA_COUNT
```

---

## Example: List Containment

### Before
```python
def test_tags():
    tags = extract_tags(article)
    assert tags is not None
    assert "ML" in tags
```

### After (with captured values: tags = ["Machine Learning", "Python", "Data Science"])
```python
EXPECTED_TAGS = ["Machine Learning", "Python", "Data Science"]

def test_tags():
    tags = extract_tags(article)
    assert tags == EXPECTED_TAGS
    # Or if order doesn't matter:
    assert set(tags) == set(EXPECTED_TAGS)
```

---

## Example: Edge Case Handling

### Before
```python
def test_empty_search():
    results = search("nonexistent")
    assert results is not None
    assert len(results) >= 0
```

### After (with captured values: results = [])
```python
def test_empty_search():
    results = search("nonexistent")
    assert results == []
    # Or:
    assert len(results) == 0
```

---

## Key Takeaways

1. **Weak assertions** capture existence/type, not actual values → Fix by using actual values
2. **Magic numbers** make assertions brittle → Fix by using named constants
3. **Complex assertions** with `and` are hard to debug → Fix by splitting into separate assertions
4. **Substring checks** are too broad → Fix by using exact matches or specific property checks
5. **Negation checks** don't verify positive requirements → Fix by checking expected positive values

Always capture the actual value from test execution to know what to assert!
