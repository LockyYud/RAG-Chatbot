import axios from 'axios';
import { Message } from '@/libs/types';
import { message, UploadFile } from 'antd';

const service = axios.create({
    withCredentials: false,
    timeout: process.env.NEXT_PUBLIC_APP_API_TIMEOUT
        ? +process.env.NEXT_PUBLIC_APP_API_TIMEOUT
        : 60000,
});

export const sendMessages = async (
    list_message: Message[] | undefined,
    files: UploadFile[] | undefined, // Change to UploadFile[]
    API_BASE_URL: string | undefined | null,
    model: string | undefined | null
) => {
    if (!API_BASE_URL) {
        message.error('API_BASE_URL is not defined');
        throw new Error('API_BASE_URL is not defined');
    }

    service.defaults.baseURL = API_BASE_URL;

    try {
        // Create FormData to send both messages and files
        const formData = new FormData();
        // Append files to FormData if any
        if (files) {
            files.forEach((file) => {
                // Ensure that the file is not a placeholder type and has a valid origin
                if (file.originFileObj) {
                    formData.append('files', file.originFileObj);
                }
            });
        }

        // Send the request
        if (files && files?.length > 0) {
            const responseFile = await service.post(`uploadfiles`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            if (responseFile.status !== 200) {
                throw new Error('Failed to upload files');
            }
        }

        if (list_message === undefined) {
            message.error('API_BASE_URL is not defined');
            throw new Error('API_BASE_URL is not defined');
        }
        const response = await service.post(`chat/${model}`, list_message[list_message.length - 1], {
            headers: { accept: 'application/json' }
        });

        return response.data; // Return the response data on success
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } catch (error: any) {
        // Check if it's an Axios error
        if (error.response) {
            // The request was made and the server responded with a status code outside of the 2xx range
            alert(`Error: ${error.response.data?.message || error.response.statusText}`);
        } else if (error.request) {
            // The request was made but no response was received
            alert('No response from the server. Please try again later.');
        } else {
            // Something else happened during the request setup
            alert(`Error: ${error.message}`);
        }

        // Rethrow the error if needed or return a default value
        throw error;
    }
};


export const getListModels = async (
    API_BASE_URL: string | undefined | null,
) => {
    if (!API_BASE_URL) {
        message.error('API_BASE_URL is not defined');
        throw new Error('API_BASE_URL is not defined');
    }

    service.defaults.baseURL = API_BASE_URL;
    const list_models = await service.get(`list_models`)
    return list_models.data;
}