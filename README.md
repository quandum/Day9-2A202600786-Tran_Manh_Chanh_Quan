# Codelab: Xây Dựng Hệ Thống Multi-Agent với A2A Protocol

**Thời gian:** 2 giờ
**Ngôn ngữ:** Python 3.11+
**Công nghệ:** LangGraph, LangChain, A2A SDK

## Mục Tiêu Học Tập

Sau khi hoàn thành codelab này, bạn sẽ:

- Hiểu cách LLM hoạt động từ cơ bản đến nâng cao
- Biết cách tích hợp tools và RAG vào LLM
- Xây dựng được single agent với ReAct pattern
- Tạo multi-agent system với LangGraph
- Triển khai distributed agents với A2A protocol

## Chuẩn Bị

### Yêu Cầu Hệ Thống

- Python 3.11 trở lên
- [uv](https://docs.astral.sh/uv/) package manager
- API key từ [OpenRouter](https://openrouter.ai)

### Cài Đặt

```bash
# Clone repository
git clone <repo-url>
cd legal_multiagent

# Cài đặt dependencies
uv sync

# Cấu hình environment
cp .env.example .env
# Sửa file .env, thêm OPENROUTER_API_KEY của bạn
```

---

## Phần 1: Direct LLM Calling (20 phút)

### Lý Thuyết

LLM (Large Language Model) ở dạng cơ bản nhất là một API nhận input text và trả về output text. Không có memory, không có tools, chỉ dựa vào training data.

**Ưu điểm:**

- Đơn giản, dễ implement
- Phản hồi nhanh

**Nhược điểm:**

- Không có kiến thức real-time
- Không thể tra cứu database
- Không có context giữa các lần gọi

### Thực Hành

**Bước 1:** Chạy demo Stage 1

```bash
uv run python stages/stage_1_direct_llm/main.py
```

**Bước 2:** Đọc và hiểu code

Mở file `stages/stage_1_direct_llm/main.py` và trả lời:

1. LLM được khởi tạo như thế nào? (Tìm hàm `get_llm()`)
2. Message được gửi đến LLM có cấu trúc gì?
3. Tại sao cần có `SystemMessage` và `HumanMessage`?

> **Trả lời:**
>
> 1. **LLM được khởi tạo qua `get_llm()`** trong `common/llm.py`. Hàm trả về một `ChatOpenAI` trỏ tới OpenRouter (OpenAI-compatible API): nó đọc model từ biến môi trường `OPENROUTER_MODEL` (mặc định `google/gemma-4-31b-it:free`), API key từ `OPENROUTER_API_KEY`, và `openai_api_base="https://openrouter.ai/api/v1"`. Nhờ chuẩn OpenAI-compatible nên có thể đổi sang bất kỳ model nào chỉ bằng biến môi trường.
> 2. **Message là một list các message object** truyền vào `llm.ainvoke(messages)`. Ở Stage 1 list gồm `[SystemMessage(...), HumanMessage(...)]`. Mỗi message có `role` (system/user/assistant) và `content` (text). LLM nhận toàn bộ list này như một hội thoại và sinh ra một `AIMessage`.
> 3. **`SystemMessage`** thiết lập vai trò, ngữ cảnh và ràng buộc cho model (ví dụ "You are a legal expert... under 300 words") — định hình *cách* trả lời. **`HumanMessage`** chứa câu hỏi thực tế của người dùng — *nội dung* cần trả lời. Tách hai loại giúp model phân biệt rõ chỉ dẫn hệ thống với input người dùng, cho output ổn định và đúng vai trò hơn; đồng thời giảm rủi ro prompt-injection so với việc gộp tất cả vào một chuỗi.

**Bài Tập 1.1:** Thay đổi câu hỏi

Sửa biến `QUESTION` thành câu hỏi pháp lý khác (tiếng Việt hoặc tiếng Anh) và chạy lại.

> **Đã hoàn thành:** Chỉ cần sửa hằng `QUESTION` ở đầu `stages/stage_1_direct_llm/main.py`, ví dụ:
> `QUESTION = "Doanh nghiệp vi phạm hợp đồng lao động thì chịu trách nhiệm pháp lý gì?"` rồi chạy lại
> `uv run python stages/stage_1_direct_llm/main.py`. Vì LLM là stateless nên mỗi câu hỏi là một lần gọi độc lập, không có ngữ cảnh từ lần trước.

**Bài Tập 1.2:** Thêm temperature control

Thêm parameter `temperature=0.3` vào hàm `get_llm()` trong `common/llm.py` để làm output ổn định hơn.

> **Đã hoàn thành:** `get_llm()` trong [common/llm.py](common/llm.py) đã thêm tham số `temperature: float = 0.3` và truyền vào `ChatOpenAI(..., temperature=temperature)`. `temperature` thấp (0.3) làm phân phối xác suất token "nhọn" hơn → output ít ngẫu nhiên, ổn định và dễ tái lập — phù hợp cho phân tích pháp lý cần tính nhất quán. Vì mọi stage đều dùng chung `get_llm()` nên thay đổi này áp dụng cho toàn hệ thống; có thể override khi cần sáng tạo hơn, ví dụ `get_llm(temperature=0.8)`.

---

## Phần 2: LLM + RAG & Tools (30 phút)

### Lý Thuyết

**RAG (Retrieval-Augmented Generation):** Cho phép LLM tra cứu knowledge base trước khi trả lời.

**Tools:** Các function mà LLM có thể gọi để thực hiện tác vụ cụ thể (tính toán, query database, gọi API).

**Function Calling Flow:**

1. LLM nhận câu hỏi + danh sách tools
2. LLM quyết định gọi tool nào (hoặc không gọi)
3. Tool được execute, trả về kết quả
4. LLM nhận kết quả và tạo câu trả lời cuối cùng

### Thực Hành

**Bước 1:** Chạy demo Stage 2

```bash
uv run python stages/stage_2_rag_tools/main.py
```

**Bước 2:** Phân tích code

Mở `stages/stage_2_rag_tools/main.py` và tìm:

1. Hàm `@tool` decorator được dùng ở đâu?
2. `LEGAL_KNOWLEDGE` được cấu trúc như thế nào?
3. LLM được bind với tools ra sao? (Tìm `.bind_tools()`)

> **Trả lời:**
>
> 1. **`@tool` decorator** (từ `langchain_core.tools`) được đặt ngay trên các hàm `search_legal_database`, `calculate_damages` (và `check_statute_of_limitations` mới thêm). Nó biến một hàm Python thành một tool mà LLM có thể gọi: docstring trở thành mô tả tool, còn type hints của tham số trở thành JSON schema để LLM biết cần truyền argument gì.
> 2. **`LEGAL_KNOWLEDGE`** là một `list` các `dict`, mỗi entry gồm: `id` (định danh nguồn luật), `keywords` (danh sách từ khóa để so khớp), và `text` (nội dung pháp lý). Đây là một "vector store" giả lập — `search_legal_database` chấm điểm bằng số từ khóa trùng (`overlap`) giữa query và `keywords`, sắp xếp giảm dần và trả về 2 entry điểm cao nhất.
> 3. **`.bind_tools()`**: `llm_with_tools = llm.bind_tools(TOOLS)` gắn danh sách tools (kèm schema) vào LLM. Khi gọi, LLM có thể trả về `tool_calls` (tên tool + arguments) thay vì trả lời thẳng. Code tự thực thi tool, đưa kết quả về dưới dạng `ToolMessage`, rồi gọi LLM lần nữa để tạo câu trả lời cuối — đây chính là vòng lặp tool-call **thủ công** (chỉ 1 vòng), khác với ReAct tự động ở Stage 3.

**Bài Tập 2.1:** Thêm knowledge base entry

Thêm một entry mới vào `LEGAL_KNOWLEDGE` về luật lao động:

```python
{
    "id": "labor_law",
    "keywords": ["lao động", "sa thải", "hợp đồng lao động", "labor", "termination"],
    "text": (
        "Theo Bộ luật Lao động Việt Nam 2019, người sử dụng lao động có thể "
        "đơn phương chấm dứt hợp đồng trong các trường hợp: (1) người lao động "
        "thường xuyên không hoàn thành công việc; (2) bị ốm đau, tai nạn đã điều trị "
        "12 tháng chưa khỏi; (3) thiên tai, hỏa hoạn; (4) người lao động đủ tuổi nghỉ hưu."
    ),
}
```

> **Đã hoàn thành:** Entry `labor_law` đã được thêm vào cuối list `LEGAL_KNOWLEDGE` trong [stages/stage_2_rag_tools/main.py](stages/stage_2_rag_tools/main.py). Khi câu hỏi chứa từ khóa như "labor", "termination", "sa thải", `search_legal_database` sẽ tính được `overlap > 0` và trả về entry này.

**Bài Tập 2.2:** Tạo tool mới

Tạo một tool `@tool` mới tên `check_statute_of_limitations` nhận vào `case_type` (string) và trả về thời hiệu khởi kiện:

```python
@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện theo loại vụ án.
  
    Args:
        case_type: Loại vụ án (contract, tort, property)
    """
    limits = {
        "contract": "4 năm (UCC § 2-725)",
        "tort": "2-3 năm tùy bang",
        "property": "5 năm",
    }
    return limits.get(case_type.lower(), "Không xác định")
```

Thêm tool này vào danh sách tools và test.

> **Đã hoàn thành:** Tool `check_statute_of_limitations` đã được định nghĩa và thêm vào `TOOLS = [search_legal_database, calculate_damages, check_statute_of_limitations]` trong [stages/stage_2_rag_tools/main.py](stages/stage_2_rag_tools/main.py). Vì tool đã được `bind_tools`, LLM tự động nhận biết và có thể gọi nó khi câu hỏi liên quan tới thời hiệu khởi kiện (ví dụ "What is the statute of limitations for a contract claim?").

---

## Phần 3: Single Agent với ReAct (25 phút)

### Lý Thuyết

**ReAct Pattern:** Reasoning + Acting

Agent tự động lặp lại chu trình:

1. **Think:** Suy nghĩ cần làm gì
2. **Act:** Gọi tool
3. **Observe:** Nhận kết quả
4. Lặp lại cho đến khi có câu trả lời cuối cùng

LangGraph cung cấp `create_react_agent` để tự động hóa pattern này.

### Thực Hành

**Bước 1:** Chạy demo Stage 3

```bash
uv run python stages/stage_3_single_agent/main.py
```

**Bước 2:** Quan sát output

Chú ý cách agent tự động:

- Quyết định tool nào cần gọi
- Gọi nhiều tools liên tiếp
- Tổng hợp kết quả

**Bước 3:** Đọc code

Mở `stages/stage_3_single_agent/main.py`:

1. Tìm `create_react_agent()` — đây là magic function
2. So sánh với Stage 2: không còn manual tool loop
3. Xem `agent_executor.invoke()` — chỉ cần gọi một lần

> **Trả lời:**
>
> 1. **`create_react_agent(model=llm, tools=TOOLS, prompt=SYSTEM_PROMPT)`** (từ `langgraph.prebuilt`) dựng sẵn một StateGraph thực hiện vòng lặp ReAct: model node → tool node → quay lại model, lặp đến khi không còn `tool_calls`. Đây là "magic" vì toàn bộ logic Think→Act→Observe được đóng gói trong một dòng.
> 2. **So với Stage 2:** Stage 2 phải tự viết vòng lặp (gọi LLM → đọc `tool_calls` → thực thi tool → append `ToolMessage` → gọi lại) và chỉ chạy **một vòng**. Stage 3 không còn đoạn code thủ công đó; agent tự quyết định gọi bao nhiêu tool và lặp bao nhiêu lần (multi-step) cho tới khi đủ thông tin.
> 3. **Một lần gọi:** chỉ cần `graph.astream(inputs, ...)` (hoặc `graph.ainvoke(inputs)`) một lần với câu hỏi; agent tự lo phần điều phối. Trong file này dùng `astream` với `stream_mode="updates"` để in ra từng bước THINK/ACT/OBSERVE cho dễ quan sát.

**Bài Tập 3.1:** Thêm tool tra cứu án lệ

```python
@tool
def search_case_law(keywords: str) -> str:
    """Tìm kiếm án lệ theo từ khóa.
  
    Args:
        keywords: Từ khóa tìm kiếm
    """
    cases = {
        "breach": "Hadley v. Baxendale (1854) - Consequential damages",
        "negligence": "Donoghue v. Stevenson (1932) - Duty of care",
        "contract": "Carlill v. Carbolic Smoke Ball Co (1893) - Unilateral contract",
    }
    for key, case in cases.items():
        if key in keywords.lower():
            return case
    return "Không tìm thấy án lệ phù hợp"
```

Thêm vào tools list và test với câu hỏi về breach of contract.

> **Đã hoàn thành:** Tool `search_case_law` đã được thêm và đăng ký vào `TOOLS` trong [stages/stage_3_single_agent/main.py](stages/stage_3_single_agent/main.py). Khi hỏi về "breach of contract", agent sẽ tự gọi tool này (THINK→ACT) và nhận về án lệ `Hadley v. Baxendale (1854)`.

**Bài Tập 3.2:** Debug agent reasoning

Thêm `verbose=True` vào `create_react_agent()` để xem chi tiết quá trình suy nghĩ của agent.

> **Trả lời / Đã hoàn thành:** Trong các phiên bản LangGraph hiện tại, `create_react_agent()` **không có** tham số `verbose=True` (đó là API cũ của `AgentExecutor` trong LangChain). Cách "debug reasoning" tương đương — và đã được dùng sẵn trong file này — là **stream các update** của graph:
>
> ```python
> async for chunk in graph.astream(inputs, stream_mode="updates"):
>     for node_name, update in chunk.items():
>         # in ra THINK + ACT (tool_calls), OBSERVE (tool result), FINAL ANSWER
> ```
>
> Cách này hiển thị từng node, từng `tool_call` (tên + args) và kết quả tool — chính là chuỗi suy luận Think→Act→Observe của agent. Ngoài ra có thể bật `debug=True` khi `create_react_agent(..., debug=True)` hoặc đặt `LANGCHAIN_TRACING_V2=true` (LangSmith) để xem trace chi tiết hơn.

---

## Phần 4: Multi-Agent In-Process (30 phút)

### Lý Thuyết

**Multi-Agent System:** Nhiều agents chuyên môn hóa cùng làm việc.

**Ưu điểm:**

- Mỗi agent tập trung vào domain riêng
- Có thể chạy song song (parallel execution)
- Dễ maintain và mở rộng

**LangGraph StateGraph:**

- Định nghĩa state (dữ liệu chia sẻ giữa các nodes)
- Tạo nodes (các bước xử lý)
- Định nghĩa edges (luồng điều khiển)

**Send API:** Cho phép dispatch nhiều tasks song song.

### Thực Hành

**Bước 1:** Chạy demo Stage 4

```bash
uv run python stages/stage_4_milti_agent/main.py
```

**Bước 2:** Phân tích kiến trúc

Mở `stages/stage_4_milti_agent/main.py`:

1. Tìm `class State(TypedDict)` — đây là shared state
2. Tìm các agent functions: `law_agent`, `tax_agent`, `compliance_agent`
3. Tìm `Send()` API — dispatch parallel tasks
4. Xem `graph.add_node()` và `graph.add_edge()`

> **Trả lời:**
>
> 1. **Shared state** là `class LegalState(TypedDict)` — chứa các field dùng chung giữa các node: `question`, `law_analysis`, các cờ `needs_tax/needs_compliance/needs_privacy`, và các kết quả `tax_result/compliance_result/privacy_result`, `final_answer`. Các field chạy song song được `Annotated[str, _last_wins]` để hai nhánh parallel cùng ghi không bị xung đột (reducer `_last_wins` giữ giá trị mới nhất).
> 2. **Các agent function** trong file này tên là: `analyze_law` (lead attorney), `call_tax_specialist`, `call_compliance_specialist` (và `call_privacy_specialist` mới thêm), cùng node điều phối `check_routing` và `aggregate`. Mỗi specialist là một `create_react_agent` riêng với system prompt và tool chuyên môn.
> 3. **`Send()` API** (từ `langgraph.constants`) xuất hiện trong `route_to_specialists`: hàm trả về một `list[Send]`, mỗi `Send("ten_node", state)` dispatch một task tới node tương ứng. LangGraph chạy các `Send` này **song song** → tax + compliance (+ privacy) chạy đồng thời.
> 4. **`graph.add_node()`** đăng ký từng bước xử lý (analyze_law, check_routing, các specialist, aggregate). **`graph.add_edge()`** định nghĩa luồng tuần tự (vd `analyze_law → check_routing`, `call_tax_specialist → aggregate`), còn `graph.add_conditional_edges("check_routing", route_to_specialists, [...])` định nghĩa nhánh động dựa trên kết quả routing.

**Bước 3:** Vẽ graph

```python
# Thêm vào cuối file main.py
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

**Bài Tập 4.1:** Thêm agent mới

Tạo `privacy_agent` chuyên về GDPR và privacy law:

```python
def privacy_agent(state: State) -> dict:
    """Agent chuyên về luật bảo vệ dữ liệu cá nhân."""
    llm = get_llm()
  
    prompt = f"""Bạn là chuyên gia về GDPR và luật bảo vệ dữ liệu cá nhân.
  
Câu hỏi gốc: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Hãy phân tích các vấn đề về privacy và GDPR (nếu có).
"""
  
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"privacy_analysis": response.content}
```

Thêm node này vào graph và kết nối với `aggregate_results`.

> **Đã hoàn thành:** Đã thêm specialist privacy vào [stages/stage_4_milti_agent/main.py](stages/stage_4_milti_agent/main.py) (đặt tên `call_privacy_specialist` cho nhất quán với các node khác trong file):
>
> - Thêm field `needs_privacy: bool` và `privacy_result: Annotated[str, _last_wins]` vào `LegalState`.
> - Thêm hàm `call_privacy_specialist(state)` dùng prompt GDPR/CCPA như đề bài.
> - Đăng ký node + edge `call_privacy_specialist → aggregate`, và `aggregate` đã gộp thêm mục `## Data Privacy Analysis`.

**Bài Tập 4.2:** Implement conditional routing

Sửa `check_routing` để chỉ gọi privacy_agent khi câu hỏi có từ khóa "data", "privacy", "gdpr":

```python
def check_routing(state: State) -> list[Send]:
    question_lower = state["question"].lower()
    tasks = []
  
    if any(kw in question_lower for kw in ["tax", "irs", "thuế"]):
        tasks.append(Send("tax_agent", state))
  
    if any(kw in question_lower for kw in ["compliance", "sec", "regulation"]):
        tasks.append(Send("compliance_agent", state))
  
    if any(kw in question_lower for kw in ["data", "privacy", "gdpr", "dữ liệu"]):
        tasks.append(Send("privacy_agent", state))
  
    return tasks if tasks else [Send("aggregate_results", state)]
```

> **Đã hoàn thành:** Routing có điều kiện đã được hiện thực trong [stages/stage_4_milti_agent/main.py](stages/stage_4_milti_agent/main.py). Vì file tách `check_routing` (đặt cờ) và `route_to_specialists` (trả về `list[Send]`), logic keyword được đặt trong `check_routing`:
> `needs_privacy = any(kw in question_lower for kw in ["data", "privacy", "gdpr", "dữ liệu"])`,
> còn `route_to_specialists` chỉ `Send("call_privacy_specialist", state)` khi `needs_privacy` là `True`. Nhờ vậy privacy specialist chỉ chạy khi câu hỏi thực sự liên quan dữ liệu/GDPR, tiết kiệm chi phí gọi LLM.

---

## Phần 5: Distributed A2A System (15 phút)

### Lý Thuyết

**A2A (Agent-to-Agent) Protocol:** Chuẩn giao tiếp giữa các agents qua HTTP.

**Khác biệt với Stage 4:**

- Mỗi agent là một service độc lập
- Giao tiếp qua HTTP thay vì in-process
- Dynamic discovery qua Registry
- Có thể scale từng agent riêng biệt

**Kiến trúc:**

```
Registry (10000) ← agents register on startup
    ↓
Customer Agent (10100) → Law Agent (10101)
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
            Tax Agent (10102)   Compliance Agent (10103)
```

### Thực Hành

**Bước 1:** Khởi động toàn bộ hệ thống

```bash
./start_all.sh
```

Chờ ~10 giây để tất cả services khởi động.

**Bước 2:** Test hệ thống

```bash
uv run python test_client.py
```

**Bước 3:** Quan sát logs

Mở 5 terminal tabs và xem logs của từng service:

- Registry: port 10000
- Customer Agent: port 10100
- Law Agent: port 10101
- Tax Agent: port 10102
- Compliance Agent: port 10103

**Bài Tập 5.1:** Trace request flow

Trong logs, tìm `trace_id` và theo dõi request đi qua các agents. Vẽ sequence diagram.

> **Trả lời:** Mỗi request được gắn một `trace_id` (sinh ở Customer Agent) và được truyền nguyên vẹn qua metadata của message A2A (`metadata={"trace_id": ...}` trong `common/a2a_client.py`). Grep cùng `trace_id` trên cả 5 service sẽ thấy đường đi: Customer → Law → (song song) Tax + Compliance → trả ngược về. Sequence diagram:
>
> ```mermaid
> sequenceDiagram
>     participant C as test_client
>     participant CU as Customer (10100)
>     participant R as Registry (10000)
>     participant L as Law (10101)
>     participant T as Tax (10102)
>     participant CO as Compliance (10103)
>     C->>CU: send_message(question)
>     CU->>R: discover("law_question")
>     R-->>CU: endpoint Law
>     CU->>L: delegate(question, trace_id, depth=1)
>     L->>L: analyze_law + check_routing
>     par parallel (Send API)
>         L->>R: discover("tax_question")
>         L->>T: delegate(depth=2)
>         T-->>L: tax_result
>     and
>         L->>R: discover("compliance_question")
>         L->>CO: delegate(depth=2)
>         CO-->>L: compliance_result
>     end
>     L->>L: aggregate
>     L-->>CU: final_answer
>     CU-->>C: response
> ```

**Bài Tập 5.2:** Test dynamic discovery

1. Dừng Tax Agent (Ctrl+C)
2. Chạy lại `test_client.py`
3. Quan sát lỗi và cách hệ thống xử lý

> **Trả lời:** Khi Tax Agent dừng, có hai khả năng tùy thời điểm:
>
> - **Nếu Tax Agent đã unregister/registry không trả endpoint:** `discover("tax_question")` thất bại → nhánh `call_tax` ném exception và được bắt trong `try/except`, trả về `"[Tax analysis unavailable: ...]"`. Hệ thống **không sập** — Compliance vẫn chạy và `aggregate` vẫn tổng hợp được câu trả lời (chỉ thiếu phần thuế).
> - **Nếu registry vẫn còn endpoint cũ (stale):** việc gọi HTTP tới Tax Agent sẽ lỗi connection refused; cũng bị `except` bắt và trả về cùng thông báo unavailable.
>
> Đây chính là tính **fault tolerance**: lỗi của một agent được cô lập (graceful degradation), không kéo đổ toàn hệ thống. Dynamic discovery cho phép registry phản ánh agent nào đang sống; nếu Tax Agent khởi động lại và register lại, lần chạy sau nó lại được dùng bình thường mà không cần sửa code.

**Bài Tập 5.3:** Modify agent behavior

Sửa `tax_agent/graph.py`, thay đổi system prompt để agent trả lời ngắn gọn hơn. Restart tax agent và test lại.

> **Trả lời / Hướng dẫn:** Mở `tax_agent/graph.py`, tìm system prompt của tax specialist và thêm ràng buộc độ dài, ví dụ thêm câu *"Trả lời thật ngắn gọn, tối đa 3 gạch đầu dòng, dưới 80 từ."* vào cuối prompt. Sau đó **restart riêng** Tax Agent (Ctrl+C tiến trình port 10102 rồi chạy lại `uv run python -m tax_agent`) và chạy lại `test_client.py`. Vì kiến trúc A2A loose-coupling, chỉ cần restart đúng service đó — Registry, Law, Compliance, Customer vẫn giữ nguyên; đây là ưu điểm về khả năng triển khai/scale độc lập so với monolith ở Stage 4.

---

## Phần 6: Tổng Kết & Mở Rộng (10 phút)

### So Sánh 5 Stages

| Stage | Pattern         | Use Case                                 | Complexity |
| ----- | --------------- | ---------------------------------------- | ---------- |
| 1     | Direct LLM      | Câu hỏi đơn giản, không cần tools | ⭐         |
| 2     | LLM + Tools     | Cần tra cứu data hoặc tính toán     | ⭐⭐       |
| 3     | ReAct Agent     | Tự động orchestration, multi-step     | ⭐⭐⭐     |
| 4     | Multi-Agent     | Nhiều domains, parallel processing      | ⭐⭐⭐⭐   |
| 5     | Distributed A2A | Production, scalable, fault-tolerant     | ⭐⭐⭐⭐⭐ |

### Câu Hỏi Ôn Tập

1. Khi nào nên dùng single agent thay vì multi-agent?
2. Ưu điểm của A2A protocol so với gRPC hoặc REST thông thường?
3. Làm thế nào để prevent infinite delegation loops trong A2A?
4. Tại sao cần Registry service? Có thể hardcode URLs không?

> **Trả lời:**
>
> 1. **Single vs multi-agent:** Dùng **single agent** khi bài toán thuộc một domain, số tool ít, độ phức tạp thấp/trung bình — đơn giản hơn, ít overhead, dễ debug, rẻ hơn (ít lần gọi LLM). Chuyển sang **multi-agent** khi cần nhiều chuyên môn khác nhau (luật + thuế + compliance), muốn chạy song song để giảm latency, hoặc cần scale/maintain từng phần độc lập. Nguyên tắc: bắt đầu đơn giản, chỉ tách agent khi prompt/tool của một agent trở nên quá tải hoặc khi cần parallel.
> 2. **A2A so với gRPC/REST thuần:** A2A xây trên HTTP nhưng **chuẩn hóa riêng cho agent**: có Agent Card (`/.well-known/agent.json`) để mô tả khả năng (skills) → cho phép **dynamic discovery**; có khái niệm `task`, `context_id`, message/parts và streaming phù hợp hội thoại nhiều bước; truyền metadata như `trace_id`, `delegation_depth` xuyên suốt. gRPC/REST chỉ là transport, bạn phải tự định nghĩa contract, discovery, định danh task... A2A đem lại tính **liên thông (interoperability)** giữa agent của các bên khác nhau mà không cần thỏa thuận API riêng.
> 3. **Chống infinite delegation loop:** Truyền `delegation_depth` trong metadata và **tăng dần mỗi lần delegate**; mỗi agent kiểm tra ngưỡng `MAX_DELEGATION_DEPTH` (trong repo = 3) — khi đạt ngưỡng thì `check_routing` trả `needs_tax=False, needs_compliance=False`, dừng việc gọi tiếp. Kết hợp với `trace_id`/`context_id` để phát hiện vòng lặp, và timeout cho mỗi HTTP call. Đây là cơ chế then chốt tránh A→B→A→B vô hạn.
> 4. **Tại sao cần Registry:** Registry cho phép **service discovery động** — agent tự đăng ký khi khởi động và client/agent khác tra cứu theo *khả năng* ("tax_question") thay vì địa chỉ cứng. Có thể hardcode URL được (đơn giản hơn cho demo), nhưng sẽ mất tính linh hoạt: không tự phát hiện agent chết/mới, khó scale nhiều instance (load-balancing), khó thay đổi cấu hình mà không sửa code và redeploy. Registry là một dạng *indirection* giúp hệ thống loose-coupled và mở rộng được trong môi trường production.

### Bài Tập Nâng Cao (Tự Học)

**Challenge 1:** Thêm memory/conversation history

Implement conversation memory để agent nhớ các câu hỏi trước đó.

**Challenge 2:** Add authentication

Thêm API key authentication cho các A2A endpoints.

**Challenge 3:** Implement retry logic

Khi một agent fail, tự động retry với exponential backoff.

**Challenge 4:** Monitoring & Observability

Tích hợp LangSmith hoặc Prometheus để monitor agent performance.

---

## Tài Liệu Tham Khảo

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [A2A Protocol Spec](https://github.com/google/A2A)
- [OpenRouter API](https://openrouter.ai/docs)
- Architecture diagrams: `docs/*.svg`

## Hỗ Trợ

Nếu gặp vấn đề:

1. Check `.env` file có đúng API key không
2. Đảm bảo tất cả ports (10000-10103) không bị chiếm
3. Xem logs trong terminal để debug
4. Đọc error messages cẩn thận — thường có hint rõ ràng

---

## **Bài Tập Cộng Điểm:**

Sau khi chạy full Stage 5 (test_client.py) trả lời 2 câu hỏi:

- Latency (Tổng thời gian trả lời 1 câu hỏi của hệ thống) là bao nhiêu giây?
- Đề xuất phương án giảm latency và demo + show thời gian xử lý đã giảm được khi apply phương án?

> **Trả lời:**
>
> **1. Đo latency:** `test_client.py` đã được bổ sung đo thời gian end-to-end (dùng `time.perf_counter()` bao quanh `client.send_message`) và in dòng `[Latency] Total end-to-end response time: X.XX seconds`. **Kết quả đo thực tế với model `google/gemma-4-31b-it:free`: `153.18 giây`** cho câu hỏi mẫu (luật + thuế + compliance). Latency cao như vậy chủ yếu do: (a) chuỗi gọi LLM tuần tự Customer → Law (`analyze_law` + `check_routing` + `aggregate`) cộng dồn; (b) Tax + Compliance tuy chạy *song song* nhưng mỗi nhánh là một ReAct agent nhiều bước; (c) model **free bị rate-limit (HTTP 429 Too Many Requests)** nên client tự retry, cộng thêm thời gian chờ. Nếu dùng model trả phí/nhanh hơn, latency sẽ giảm mạnh.
>
> **Phân tích nguồn latency:** Phần lớn thời gian là **độ trễ inference của LLM**, không phải HTTP overhead. Các node *tuần tự* (analyze_law → check_routing → ... → aggregate) cộng dồn; các specialist đã chạy *song song* nên không cộng dồn.
>
> **2. Đề xuất phương án giảm latency (kèm cách demo so sánh):**
>
> | Phương án                                   | Cách làm                                                                                                                        | Kỳ vọng                                  |
> | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
> | **Bỏ bớt node LLM thừa**              | Gộp `check_routing` bằng keyword-matching (không gọi LLM) thay vì hỏi LLM để quyết định routing                      | Tiết kiệm trọn 1 lần gọi LLM (~3–8s) |
> | **Dùng model nhỏ/nhanh cho node phụ** | Đặt `OPENROUTER_MODEL` của routing/specialist sang model nhẹ (vd `*-haiku`/`*-mini`), giữ model lớn cho `aggregate` | Giảm 30–50% thời gian mỗi node phụ    |
> | **Streaming output**                     | Stream token của `aggregate` về client thay vì chờ trọn câu trả lời                                                     | Giảm*perceived latency* rõ rệt        |
> | **Giảm token**                          | Rút gọn system prompt + giới hạn `max_tokens` cho specialist                                                                | Giảm thời gian sinh token                |
> | **Đảm bảo parallel thực sự**        | Kiểm tra Tax + Compliance chạy đồng thời qua `Send` (đã có)                                                             | Tránh cộng dồn tuần tự                |
>
> **Demo so sánh (đã triển khai):** Phương án **"bỏ node LLM thừa"** đã được code thật trong [law_agent/graph.py](law_agent/graph.py). Hàm `check_routing` giờ đọc biến môi trường `ROUTING_MODE`:
>
> - `ROUTING_MODE=llm` (baseline): gọi thêm **1 lần LLM** để hỏi nên route tới Tax/Compliance hay không.
> - `ROUTING_MODE=keyword` (tối ưu): quyết định routing bằng **keyword-matching thuần Python**, *không gọi LLM* → loại bỏ hẳn 1 round-trip LLM trên đường tuần tự.
>
> Cách chạy demo:
>
> ```powershell
> # Baseline
> $env:ROUTING_MODE="llm"; uv run python -m law_agent   # (cùng registry + tax + compliance + customer)
> uv run python test_client.py                          # đọc dòng [Latency]
>
> # Tối ưu
> $env:ROUTING_MODE="keyword"; uv run python -m law_agent
> uv run python test_client.py                          # đọc dòng [Latency] mới
> ```
> **Kết quả đo & ghi chú trung thực:**
>
> | Lần chạy             | Chế độ routing | Latency đo được | Trạng thái                                                |
> | ---------------------- | ----------------- | ------------------- | ----------------------------------------------------------- |
> | Run sạch ban đầu    | `llm`           | **153.18 s**  | ✅ Thành công (đầy đủ tax + compliance)               |
> | Re-test (cùng phiên) | `llm`           | 95.05 s             | ⚠️ Kết thúc bằng HTTP 429 (đã chạm hạn mức ngày) |
> | Re-test (cùng phiên) | `keyword`       | 4.57 s              | ⚠️ Fail-fast tại LLM call đầu của Customer (HTTP 429) |
>
> **Vì sao chưa có phép đo "thành công" sạch cho `keyword`:** API key model free `google/gemma-4-31b-it:free` có hạn mức **50 requests/ngày**, và trong lúc đo lại đã **cạn quota** (`X-RateLimit-Remaining: 0`, lỗi `429 Rate limit exceeded: free-models-per-day`). Khi quota đã hết, mọi request đều fail nên hai con số 95.05s/4.57s **không phản ánh hiệu năng thật** — chúng chỉ cho thấy mỗi đường đi chạm "bức tường rate-limit" nhanh hay chậm.
>
> **Bằng chứng phân tích từ log run sạch (153.18s):** Riêng node `check_routing` ở chế độ `llm` tốn **~27 giây** cho 1 lần gọi LLM (đo từ timestamp giữa `analyze_law` xong → `check_routing` xong trong log Law Agent). Chế độ `keyword` loại bỏ hoàn toàn lần gọi này, nên **kỳ vọng giảm ~27s (~18%)** trên đường tuần tự, đồng thời **giảm 1 request lên model free → giảm rủi ro 429**. Để có bảng *Before → After* sạch, chạy lại đúng 2 lệnh trên **sau khi quota reset** (hoặc nạp 10 credits để mở 1000 req/ngày).
