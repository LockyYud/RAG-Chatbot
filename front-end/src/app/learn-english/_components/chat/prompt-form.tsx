"use client";

import * as React from "react";
import { ChatState, Message } from "@/libs/types";
import { SendButton } from "./send-button";
import { Form, Input } from "antd";
import ButtonUploadFile from "./uploadFile";
import ListUploadFile from "./list-upload-files";
import { useChatContext } from "@/libs/context/chat-context";

export function PromptForm({
    setListMessage,
}: {
    setListMessage: React.Dispatch<React.SetStateAction<Message[]>>;
}) {
    const { setState } = useChatContext();
    const [input, setInput] = React.useState("");
    const inputRef = React.useRef<HTMLTextAreaElement>(null);
    const [form] = Form.useForm();
    React.useEffect(() => {
        if (inputRef.current) {
            inputRef.current.focus();
        }
    }, []);
    const onSubmit = () => {
        const value = input.trim();
        if (!value) return;
        const newMessage: Message = { role: "user", content: value };
        setState(ChatState.BOT_TURN);
        setListMessage((prev) => [...prev, newMessage]);
        form.resetFields();
    };
    const handleKeyDown = (
        event: React.KeyboardEvent<HTMLTextAreaElement>
    ): void => {
        if (
            event.key === "Enter" &&
            !event.shiftKey &&
            !event.nativeEvent.isComposing
        ) {
            form.submit();
        }
    };
    return (
        <Form
            form={form}
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            onFinish={onSubmit}
            className=" text-black w-full p-4 focus-within:outline-none rounded-3xl disabled:cursor-not-allowed disabled:bg-zinc-200 bg-zinc-100 relative"
            style={{ minHeight: "56px", maxHeight: "240px" }} // dynamic height
        >
            <Form.Item className="absolute my-auto -top-20">
                <ListUploadFile />
            </Form.Item>
            <Form.Item className="absolute my-auto top-0 h-full flex flex-col align-middle justify-center left-3">
                <ButtonUploadFile />
            </Form.Item>
            <Form.Item
                name="inputField"
                className="flex-grow mb-0 ml-8 mr-10" // Grow to take up available space
                style={{ marginBottom: 0 }}
            >
                <Input.TextArea
                    autoSize={{ minRows: 1, maxRows: 6 }}
                    placeholder="Send message"
                    autoComplete="off"
                    autoCorrect="off"
                    style={{
                        border: "none",
                        boxShadow: "none",
                        outline: "none",
                        fontSize: "18px",
                        resize: "none", // Prevent manual resizing
                        background: "transparent",
                    }}
                    className="h-full bg-transparent hover:bg-transparent focus:bg-transparent"
                    value={input}
                    onKeyDown={handleKeyDown}
                    onChange={(e) => setInput(e.target.value)}
                />
            </Form.Item>
            <Form.Item className="absolute my-auto top-0 h-full flex flex-col align-middle justify-center right-3">
                <SendButton state disabled={!(input.length > 0)} />
            </Form.Item>
        </Form>
    );
}
