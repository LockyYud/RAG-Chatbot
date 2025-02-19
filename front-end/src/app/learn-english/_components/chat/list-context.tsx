import React from "react";
import type { CollapseProps } from "antd";
import { Collapse } from "antd";
import { CheckCircleTwoTone } from "@ant-design/icons";

interface CorrectionMEssageProps {
    messageCorrection: string;
    correct: boolean;
}

export const CorrectionMessage: React.FC<CorrectionMEssageProps> = ({
    messageCorrection,
    correct,
}) => {
    const items: CollapseProps["items"] = [messageCorrection].map(
        (content, index) => ({
            key: `${index + 1}`, // Ensure key is a string
            label: correct ? (
                <>
                    True
                    <CheckCircleTwoTone />
                </>
            ) : (
                "False"
            ), // Use `label` for newer Ant Design
            children: <p>{content}</p>, // Use `children` for content
        })
    );

    return (
        <Collapse defaultActiveKey={correct ? [] : ["1"]} ghost items={items} />
    );
};
