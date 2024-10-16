import axios from 'axios';

const API_BASE_URL =
    // process.env.NEXT_PUBLIC_API_BASE_URL || "localhost:8080/api/v1/";
    'http://localhost:8080/api/v1/';

const service = axios.create({
    withCredentials: false,
    baseURL: API_BASE_URL,
    timeout: process.env.NEXT_PUBLIC_APP_API_TIMEOUT
        ? +process.env.NEXT_PUBLIC_APP_API_TIMEOUT
        : 60000,
    headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
    },
    httpsAgent: new (require('https').Agent)({
        rejectUnauthorized: false, // Disable SSL verification
    }),
});

service.interceptors.request.use(
    (config) => {
        config.headers['accept'] = 'application/json';
        config.headers['Content-Type'] = 'application/json';
        return config;
    },
    (error) => {
        return Promise.reject(error);
    },
);

export const gradeEssay = async (data: { content: string }) => {
    return (await service.post(`grader/run`, data)).data;
};

export const getTest = async () => {
    return (await service.get(`ai/create-lesson-plan`)).data;
};
