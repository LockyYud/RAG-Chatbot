import React from "react";
import type { CollapseProps } from "antd";
import { Collapse } from "antd";

interface ListRelatedContentProps {
    list_content: string[];
}

export const ListRelatedContent: React.FC<ListRelatedContentProps> = ({
    list_content,
}) => {
    const items: CollapseProps["items"] = list_content.map(
        (content, index) => ({
            key: `${index + 1}`, // Ensure key is a string
            label: `Đoạn văn ${index + 1}`, // Use `label` for newer Ant Design
            children: <p>{content}</p>, // Use `children` for content
        })
    );

    return <Collapse defaultActiveKey={["1"]} ghost items={items} />;
};
