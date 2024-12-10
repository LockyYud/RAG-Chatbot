import { Select, Space } from "antd";
import { SetStateAction } from "react";

interface SelectModelsProps {
    model: string;

    set_model: (value: SetStateAction<string>) => void;

    list_model: string[];
}

export default function SelectModels({
    model,
    set_model,
    list_model,
}: SelectModelsProps) {
    return (
        <Space wrap>
            <Select
                defaultValue={model}
                // style={{ width: "100%" }}
                onChange={(value: string) => {
                    set_model(value);
                }}
                options={[
                    list_model.map((model) => ({
                        value: model,
                        label: model,
                    })),
                ]}
            />
        </Space>
    );
}
