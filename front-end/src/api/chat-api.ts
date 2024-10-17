import axios from 'axios';
import { Message } from '@/libs/types';

const service = axios.create({
    withCredentials: false,
    timeout: process.env.NEXT_PUBLIC_APP_API_TIMEOUT
        ? +process.env.NEXT_PUBLIC_APP_API_TIMEOUT
        : 60000,
});

export const sendMessages = async (
    list_message: Message[] | undefined,
    API_BASE_URL: string | undefined | null,
) => {
    if (!API_BASE_URL) {
        throw new Error('API_BASE_URL is not defined');
    }
    service.defaults.baseURL = API_BASE_URL;
    return (await service.post(`chat`, list_message)).data;
};