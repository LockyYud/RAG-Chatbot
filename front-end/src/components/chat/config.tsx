"use client";
// import { getListModels } from "@/api/chat-api";
import { Button, Input, Space, Select, Flex } from "antd";
import { useEffect, useState } from "react";
export default function ConfigBE() {
    const [model, set_model] = useState("None");
    const [api_base_url, set_url] = useState("");
    // const [list_model, set_list_model] = useState<string[]>([]);
    const [save_url, set_save_url] = useState(true);
    const list_model = ["gpt-4o-mini", "gpt-4o", "llama3.1-70B"];
    useEffect(() => {
        localStorage.setItem("model", model);
        if (typeof window !== "undefined") {
            // Now safe to use localStorage
            const savedUrl = localStorage.getItem("api_base_url") || "";
            set_url(savedUrl);
        }
    }, [model]);
    // useEffect(() => {
    //     const get_list_model = async () => {
    //         await getListModels(api_base_url).then((res) => {
    //             set_list_model(Array.isArray(res) ? res : []);
    //         });
    //     };
    //     if (api_base_url != "" && api_base_url != null) {
    //         get_list_model();
    //     }
    //     // eslint-disable-next-line react-hooks/exhaustive-deps
    // }, [save_url]);
    return (
        <div className="flex-col">
            <Flex gap="middle" vertical>
                URL Backend
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
                                localStorage.setItem(
                                    "api_base_url",
                                    api_base_url
                                );
                            } else {
                                set_save_url(false);
                            }
                        }}
                    >
                        {save_url ? "Edit" : "Save"}
                    </Button>
                </Space.Compact>
                Model
                <Space wrap>
                    <Select
                        defaultValue={model}
                        style={{ width: "100%" }}
                        onChange={(value: string) => {
                            set_model(value);
                        }}
                        options={(list_model || []).map((model) => ({
                            value: model,
                            label: model,
                        }))}
                    />
                </Space>
            </Flex>
        </div>
    );
}
