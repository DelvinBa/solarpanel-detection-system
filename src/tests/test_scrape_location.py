"""
Tests for the scrape_location module.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the src directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webscraping.scrape_location import get_coordinates


class TestScrapeLocation(unittest.TestCase):
    """Test cases for the scrape_location module."""

    @patch('webscraping.scrape_location.requests.get')
    def test_get_coordinates_success(self, mock_get):
        """Test successful coordinate retrieval."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "verblijfsobject": {
                "geometrie": {
                    "punt": {
                        "coordinates": [123456.789, 987654.321, 0.0]
                    }
                }
            }
        }
        mock_get.return_value = mock_response

        # Call the function
        result = get_coordinates("0123456789")

        # Check if the function returned the expected coordinates
        self.assertEqual(result, [123456.789, 987654.321])

    @patch('webscraping.scrape_location.requests.get')
    def test_get_coordinates_api_error(self, mock_get):
        """Test handling of API errors."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        # Call the function
        result = get_coordinates("0123456789")

        # Check if the function returns None on API error
        self.assertIsNone(result)

    @patch('webscraping.scrape_location.requests.get')
    def test_get_coordinates_missing_data(self, mock_get):
        """Test handling of missing data in response."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"verblijfsobject": {}}
        mock_get.return_value = mock_response

        # Call the function
        result = get_coordinates("0123456789")

        # Check if the function returns None when data is missing
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main() 