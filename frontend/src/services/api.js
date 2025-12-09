import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: `${API_BASE_URL}/api/v1`,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const analyzeNews = async (text) => {
  const response = await apiClient.post('/analyze', { text });
  return response.data;
};

export const submitFeedback = async (feedbackData) => {
  const response = await apiClient.post('/feedback', feedbackData);
  return response.data;
};

export default apiClient;
