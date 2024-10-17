import React from 'react';
import { PromptForm } from './prompt-form';
import { ChatState, Message } from '@/libs/types';
import useSpeechRecognize from '@/libs/hooks/use-speech-recognition';
import { RecordButton, TextRecord } from './recorder';
import { useChatContext } from '@/libs/context/chat-context';
export interface ChatPanelProps {
    setListMessage: React.Dispatch<React.SetStateAction<Message[]>>;
    isAtBottom: boolean;
    scrollToBottom: () => void;
}
export function ChatPanel({
    setListMessage,
}: ChatPanelProps) {
    const { text, cancelVoice, startListen, stopListen, isListen } =
        useSpeechRecognize();
    const { state } = useChatContext();
    return (
        <div className="fixed inset-x-0 bottom-0 w-full bg-background">
            <div className="mx-auto sm:max-w-2xl ">
                {text && (
                    <TextRecord
                        text={text}
                        funcSend={() => {
                            setListMessage((prev) => [
                                ...prev,
                                { role: 'user', content: text },
                            ]);
                            cancelVoice();
                        }}
                        funcCancel={cancelVoice}
                    />
                )}
                <div className="relative w-full">
                    <RecordButton
                        state={isListen}
                        startFunc={startListen}
                        stopFunc={stopListen}
                        disabled={state !== ChatState.USER_TURN}
                    />
                    <PromptForm setListMessage={setListMessage} />
                </div>
            </div>
            <div className="text-center w-screen text-gray-400">
                By messaging based Llama3. So Chatbot can make mistakes. Please
                verify important information.
            </div>
        </div>
    );
}
