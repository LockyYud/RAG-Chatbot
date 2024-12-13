import { BotMessage } from "@/components/chat/message";
import MarkdownRenderer from "@/components/MarkdownRender/markdown";

export default function Page() {
    const text = `Câu trả lời đúng cho câu hỏi là: **C. Tạo cơ sở cho sự hình thành phong trào dân tộc theo khuynh hướng vô sản.**\n\nGiải thích:\n1. **C. Tạo cơ sở cho sự hình thành phong trào dân tộc theo khuynh hướng vô sản.** - Điều này được thể hiện rõ trong đoạn văn 2, nơi nhấn mạnh rằng Nguyễn Ái Quốc đã truyền bá chủ nghĩa Mác - Lênin, làm chuyển biến phong trào yêu nước chống Pháp từ lập trường tiểu tư sản sang lập trường vô sản. Những hoạt động và tư tưởng của Nguyễn Ái Quốc trong những năm 20 của thế kỷ XX đã tạo nền tảng vững chắc cho sự hình thành của phong trào dân tộc theo khuynh hướng vô sản.\n\n2. **Lựa chọn A: Làm cho phong trào yêu nước chuyển hẳn sang quỹ đạo cách mạng vô sản.** - Đây không hoàn toàn chính xác vì mặc dù Nguyễn Ái Quốc đã tác động mạnh mẽ đến phong trào yêu nước, nhưng \"chuyển hẳn\" có thể không phản ánh chính xác từng bước chuyển biến mà cần có thời gian và nhiều yếu tố khác.\n\n3. **Lựa chọn B: Chấm dứt tình trạng khủng hoảng về đường lối cứu nước đầu thế kỉ XX.** - Mặc dù Nguyễn Ái Quốc đã góp phần làm rõ đường lối cách mạng, nhưng không có thông tin cụ thể trong các đoạn văn cho thấy rằng ông hoàn toàn chấm dứt khủng hoảng đó. Đường lối cứu nước vẫn cần thời gian và cuộc cách mạng tiếp theo để trở nên rõ ràng và ổn định.\n\n4. **Lựa chọn D: Trực tiếp chuẩn bị đầy đủ những điều kiện cho sự ra đời của Đảng Cộng sản.** - Tuy Nguyễn Ái Quốc đã có những đóng góp lớn cho phong trào cộng sản, nhưng sự ra đời của Đảng Cộng sản Việt Nam còn phụ thuộc vào nhiều yếu tố khác từ nhiều cá nhân và tổ chức khác nhau. Lựa chọn này thể hiện một khía cạnh quan trọng nhưng không hoàn toàn chính xác so với lựa chọn C.\n\nVì những lý do trên, lựa chọn C được xem là câu trả lời đúng nhất trong bối cảnh đã được cung cấp."
`;
    return (
        <div>
            <BotMessage>
                <MarkdownRenderer markdownText={text} />
            </BotMessage>
        </div>
    );
}
