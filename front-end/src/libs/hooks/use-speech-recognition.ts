'use client';
import { useEffect, useState } from 'react';

declare global {
    interface Window {
        webkitSpeechRecognition: unknown;
    }

    interface SpeechRecognition {
        continuous: boolean;
        lang: string;
        start(): void;
        stop(): void;
        onresult: ((event: SpeechRecognitionEvent) => void) | null;
    }

    interface SpeechRecognitionEvent {
        results: {
            [key: number]: {
                [key: number]: {
                    transcript: string;
                };
            };
        };
    }
}

const useSpeechRecognize = () => {
    const [text, setText] = useState('');
    const [isListen, setIsListen] = useState(false);
    const [recognition, setRecognition] = useState<SpeechRecognition | null>(
        null,
    );
    useEffect(() => {
        if (
            !recognition &&
            typeof window !== 'undefined' &&
            'webkitSpeechRecognition' in window
        ) {
            const recognitionInstance = new (
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                window as any
            ).webkitSpeechRecognition();
            setRecognition(recognitionInstance);
        }

        if (!recognition) return;
        recognition.continuous = true;
        recognition.lang = 'en-US';
        const handleResult = (event: SpeechRecognitionEvent) => {
            const transcript = event.results[0][0].transcript;
            setText(transcript);
        };

        recognition.onresult = handleResult;

        return () => {
            recognition.onresult = null;
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [recognition]);

    const startListen = () => {
        setText('');
        setIsListen(true);
        recognition?.start();
    };

    const stopListen = () => {
        setIsListen(false);
        recognition?.stop();
    };

    const cancelVoice = () => {
        setText('');
    };
    return {
        text,
        isListen,
        startListen,
        stopListen,
        cancelVoice,
        hasRecognitionSupport: !!recognition,
    };
};

export default useSpeechRecognize;
