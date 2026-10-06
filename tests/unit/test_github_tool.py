"""Unit tests for GitHubTool, including its hand-off to RepoAnalyzer (issue #18)."""

import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from agent.tools.github_tool import GitHubTool
from ingestion.parsers.repo_analyzer import RepoAnalyzer

pytestmark = pytest.mark.unit

API = "https://api.github.com"
REPO_URL = f"{API}/repos/octocat/demo"

REPO_JSON: dict[str, Any] = {
    "name": "demo",
    "description": "Demo repository",
    "language": "Python",
    "stargazers_count": 3,
    "forks_count": 1,
    "open_issues_count": 0,
    "pushed_at": "2026-01-01T00:00:00Z",
    "topics": [],
    "homepage": None,
    "default_branch": "main",
}

TREE_JSON: dict[str, Any] = {
    "sha": "abc123",
    "truncated": False,
    "tree": [
        {"path": "README.md", "type": "blob"},
        {"path": "src", "type": "tree"},
        {"path": "src/app.py", "type": "blob"},
        {"path": "tests", "type": "tree"},
        {"path": "tests/test_example.py", "type": "blob"},
        {"path": ".github", "type": "tree"},
        {"path": ".github/workflows", "type": "tree"},
        {"path": ".github/workflows/ci.yml", "type": "blob"},
    ],
}


def _response(url: str, status: int, payload: dict | None = None) -> httpx.Response:
    return httpx.Response(status, json=payload, request=httpx.Request("GET", url))


def _fake_get(url: str, **kwargs) -> httpx.Response:
    """Stand in for the GitHub API over HTTP."""
    if url == REPO_URL:
        return _response(url, 200, REPO_JSON)
    if "/git/trees/" in url:
        return _response(url, 200, TREE_JSON)
    raise AssertionError(f"unexpected GET {url}")


def _fake_head(url: str, **kwargs) -> httpx.Response:
    return _response(url, 200)


class TestGitHubToolFileStructure:
    def test_tool_output_feeds_analyzer_test_and_ci_detection(self):
        """The tool's own output must let RepoAnalyzer see tests and CI."""
        with (
            patch("httpx.get", side_effect=_fake_get),
            patch("httpx.head", side_effect=_fake_head),
        ):
            result = GitHubTool().execute({"github_username": "octocat", "repo_name": "demo"})

        assert result.success is True
        assert "file_structure" in result.data
        assert "tests/test_example.py" in result.data["file_structure"]
        assert ".github/workflows/ci.yml" in result.data["file_structure"]

        parsed = RepoAnalyzer().parse(json.dumps(result.data))

        assert parsed.metadata["has_tests"] is True
        assert parsed.metadata["has_ci"] is True


class FakeGitHub:
    """Strict stand-in for the GitHub HTTP boundary that records every request."""

    def __init__(self, repo=None, tree=None, tree_status=200, tree_error=None):
        self.repo = REPO_JSON if repo is None else repo
        self.tree = TREE_JSON if tree is None else tree
        self.tree_status = tree_status
        self.tree_error = tree_error
        self.gets: list[tuple[str, dict]] = []

    def get(self, url: str, **kwargs) -> httpx.Response:
        self.gets.append((url, kwargs))
        if url == REPO_URL:
            return _response(url, 200, self.repo)
        if "/git/trees/" in url:
            if self.tree_error is not None:
                raise self.tree_error
            return _response(url, self.tree_status, self.tree)
        raise AssertionError(f"unexpected GET {url}")

    def tree_requests(self) -> list[tuple[str, dict]]:
        return [call for call in self.gets if "/git/trees/" in call[0]]

    def execute(self, token: str | None = None):
        with patch("httpx.get", side_effect=self.get), patch("httpx.head", side_effect=_fake_head):
            return GitHubTool(api_token=token).execute(
                {"github_username": "octocat", "repo_name": "demo"}
            )


def _tree(*paths: str) -> dict:
    return {"truncated": False, "tree": [{"path": p, "type": "blob"} for p in paths]}


class TestFileStructureDetection:
    def test_tool_result_keeps_existing_metadata_and_adds_file_structure(self):
        result = FakeGitHub().execute()

        assert result.success is True
        assert result.data.pop("file_structure") == [e["path"] for e in TREE_JSON["tree"]]
        assert result.data == {
            "name": "demo",
            "description": "Demo repository",
            "primary_language": "Python",
            "star_count": 3,
            "fork_count": 1,
            "open_issues_count": 0,
            "last_commit_date": "2026-01-01T00:00:00Z",
            "has_readme": True,
            "topics": [],
            "homepage": "",
        }

    def test_repo_without_tests_or_ci_reports_false(self):
        result = FakeGitHub(tree=_tree("README.md", "src/app.py")).execute()
        parsed = RepoAnalyzer().parse(json.dumps(result.data))

        assert result.success is True
        assert result.data["file_structure"] == ["README.md", "src/app.py"]
        assert parsed.metadata["has_tests"] is False
        assert parsed.metadata["has_ci"] is False

    def test_valid_empty_tree_returns_empty_list(self):
        result = FakeGitHub(tree=_tree()).execute()
        parsed = RepoAnalyzer().parse(json.dumps(result.data))

        assert result.success is True
        assert result.data["file_structure"] == []
        assert parsed.metadata["has_tests"] is False
        assert parsed.metadata["has_ci"] is False


class TestTreeRequest:
    def test_requests_default_branch_recursively_with_finite_timeout(self):
        fake = FakeGitHub()
        fake.execute()

        assert len(fake.gets) == 2
        ((url, kwargs),) = fake.tree_requests()
        assert url == f"{API}/repos/octocat/demo/git/trees/main"
        assert kwargs["params"] == {"recursive": "1"}
        assert 0 < kwargs["timeout"] < float("inf")

    def test_non_main_branch_with_slash_is_url_encoded(self):
        fake = FakeGitHub(repo={**REPO_JSON, "default_branch": "release/v1 beta"})
        result = fake.execute()

        ((url, _),) = fake.tree_requests()
        assert result.success is True
        assert url == f"{API}/repos/octocat/demo/git/trees/release%2Fv1%20beta"

    def test_token_is_sent_to_tree_request(self):
        fake = FakeGitHub()
        fake.execute(token="secret-token")

        ((_, kwargs),) = fake.tree_requests()
        assert kwargs["headers"] == {"Authorization": "token secret-token"}

    def test_authorization_is_omitted_without_token(self):
        fake = FakeGitHub()
        fake.execute()

        ((_, kwargs),) = fake.tree_requests()
        assert "Authorization" not in kwargs["headers"]


class TestRetrievalFailures:
    @pytest.mark.parametrize(
        ("status", "error"),
        [
            (404, "Repository not found"),
            (403, "Rate limited or access denied"),
            (500, "GitHub API error: 500"),
        ],
    )
    def test_http_error_on_tree_request_fails(self, status, error):
        result = FakeGitHub(tree={}, tree_status=status).execute()

        assert result.success is False
        assert result.data == {}
        assert result.error == error

    def test_timeout_on_tree_request_fails(self):
        result = FakeGitHub(tree_error=httpx.ReadTimeout("timed out")).execute()

        assert result.success is False
        assert result.data == {}
        assert result.error == "timed out"

    @pytest.mark.parametrize(
        "tree",
        [
            [],
            {"truncated": False},
            {"truncated": False, "tree": "src/app.py"},
            {"truncated": False, "tree": ["src/app.py"]},
            {"truncated": False, "tree": [{"type": "blob"}]},
            {"truncated": False, "tree": [{"path": 123}]},
            {"truncated": False, "tree": [{"path": ""}]},
            {"tree": [{"path": "src/app.py"}]},
        ],
    )
    def test_malformed_tree_fails(self, tree):
        result = FakeGitHub(tree=tree).execute()

        assert result.success is False
        assert result.data == {}
        assert result.error

    def test_truncated_tree_fails_instead_of_reporting_partial_paths(self):
        result = FakeGitHub(tree={**_tree("README.md"), "truncated": True}).execute()

        assert result.success is False
        assert result.data == {}
        assert "truncated" in result.error

    @pytest.mark.parametrize("branch", [None, ""])
    def test_missing_default_branch_fails_without_tree_request(self, branch):
        fake = FakeGitHub(repo={**REPO_JSON, "default_branch": branch})
        result = fake.execute()

        assert result.success is False
        assert result.data == {}
        assert fake.tree_requests() == []
