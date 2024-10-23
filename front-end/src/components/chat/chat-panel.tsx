import React from "react";
import { PromptForm } from "./prompt-form";
import { Message } from "@/libs/types";
// import useSpeechRecognize from "@/libs/hooks/use-speech-recognition";
// import { RecordButton, TextRecord } from "./recorder";
// import { useChatContext } from "@/libs/context/chat-context";
export interface ChatPanelProps {
    setListMessage: React.Dispatch<React.SetStateAction<Message[]>>;
    isAtBottom: boolean;
    scrollToBottom: () => void;
}
export function ChatPanel({ setListMessage }: ChatPanelProps) {
    return (
        <div className="bg-background w-full min-w-fit">
            <div className="mx-auto sm:max-w-3xl ">
                <PromptForm setListMessage={setListMessage} />
            </div>
        </div>
    );
}
