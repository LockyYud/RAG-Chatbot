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
    // const { text, cancelVoice, startListen, stopListen, isListen } =
    //     useSpeechRecognize();
    // const { state } = useChatContext();
    return (
        <div className="bottom-0 bg-background absolute w-full min-w-fit">
            <div className="mx-auto sm:max-w-2xl ">
                {/* {text && (
                    <TextRecord
                        text={text}
                        funcSend={() => {
                            setListMessage((prev) => [
                                ...prev,
                                { role: "user", content: text },
                            ]);
                            cancelVoice();
                        }}
                        funcCancel={cancelVoice}
                    />
                )} */}
                {/* <RecordButton
                        state={isListen}
                        startFunc={startListen}
                        stopFunc={stopListen}
                        disabled={state !== ChatState.USER_TURN}
                    /> */}
                <PromptForm setListMessage={setListMessage} />
            </div>
        </div>
    );
}
