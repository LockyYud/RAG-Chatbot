import axios from 'axios';
import { Message } from '@/lib/types';
import { message } from 'antd';
// import { toast } from "sonner";
// import { AUTH, HTTP_STATUS } from "@/lib/constant/constant";
// import { removeCookie } from "@/lib/utils";

const API_BASE_URL =
    process.env.NEXT_PUBLIC_API_BASE_URL || 'https://bot.lingobee.vn/';

const service = axios.create({
    withCredentials: false,
    baseURL: API_BASE_URL,
    timeout: process.env.NEXT_PUBLIC_APP_API_TIMEOUT
        ? +process.env.NEXT_PUBLIC_APP_API_TIMEOUT
        : 60000,
});

service.interceptors.request.use(
    (config: any) => {
        const access_token =
            'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjp7InVzZXJJZCI6IjY2YzczNTkyMTNjNTZhNGRiODFjMjJhOSIsInJvbGUiOiJBZG1pbiJ9LCJpYXQiOjE3MjQ1OTY0ODAsImV4cCI6MTcyNTQ2MDQ4MH0.5IWUFS4f4VkdfZ3uWyGdNUKqrzla1X2mMjLt5VDQaAw';
        config.headers['Authorization'] = 'Bearer ' + access_token;
        config.headers['accept'] = 'application/json';
        config.headers['Content-Type'] = 'application/json';
        return config;
    },
    (error: any) => {
        return Promise.reject(error);
    },
);

export const sendMessage = async (
    conversationId: string,
    message: Message | undefined,
) => {
    return (await service.post(`chat/${conversationId}`, message)).data;
};

export const getMessage = async (conversationId: string) => {
    return (await service.get(`chat/message/${conversationId}`)).data;
};

export const createConversation = async (topic: string, grade: number) => {
    return (await service.get(`chat/create?topic=${topic}&grade=${grade}`))
        .data;
};

export const deleteConversation = async (conversationId: string) => {
    return (await service.delete(`chat/delete/${conversationId}`)).data;
};

export const suggestMessage = async (conversationId: string) => {
    return (
        await service.get(`chat/suggest/${conversationId}?num_suggestions=2`)
    ).data;
};

export const translateEn2Vn = async (content: string) => {
    return (await service.post(`/chat/translate/en-vn`, { message: content }))
        .data;
};
