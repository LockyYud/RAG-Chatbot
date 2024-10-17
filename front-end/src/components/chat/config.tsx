"use client";
import { sendMessages } from "@/api/chat-api";
import { Message } from "@/libs/types";
import { Button, Input, Space } from "antd";
import { useEffect, useState } from "react";

export default function ConfigBE() {
    const [api_base_url, set_url] = useState("");
    useEffect(() => {
        if (typeof window !== "undefined") {
            // Now safe to use localStorage
            const savedUrl = localStorage.getItem("api_base_url") || "";
            set_url(savedUrl);
        }
    }, []);
    const [save_url, set_save_url] = useState(true);
    return (
        <div className="flex-col gap-4">
            <h1>Config</h1>
            <Space.Compact style={{ width: "100%" }}>
                <Input
                    placeholder="Ngrok url"
                    onChange={(e) => {
                        set_url(e.target.value);
                    }}
                    defaultValue={api_base_url}
                    disabled={save_url}
                />
                <Button
                    type="primary"
                    onClick={() => {
                        if (!save_url) {
                            set_save_url(true);
                            localStorage.setItem("api_base_url", api_base_url);
                        } else {
                            set_save_url(false);
                        }
                    }}
                >
                    {save_url ? "Edit" : "Save"}
                </Button>
            </Space.Compact>
            <Button
                onClick={() => {
                    const listMessage: Message[] = [
                        { role: "user", content: "Thủ đô của Việt Nam là" },
                    ];
                    sendMessages(listMessage, api_base_url);
                }}
            >
                Test
            </Button>
            {/* <Space.Compact style={{ width: "100%" }}>
                <Input />
                <Button type="primary">Submit</Button>
            </Space.Compact>
            <Space.Compact style={{ width: "100%" }}>
                <Input />
                <Button type="primary">Submit</Button>
            </Space.Compact> */}
        </div>
    );
}
