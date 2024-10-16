import axios from 'axios';

const BASE_API =
    process.env.BASE_API || 'https://saokhue-ai-test-61nm.onrender.com/api/v1';

const request = axios.create({
    withCredentials: false,
    baseURL: BASE_API,
    timeout: process.env.APP_API_TIMEOUT ? +process.env.APP_API_TIMEOUT : 60000,
});

export default request;
