require("chatgpt").setup {
  --   welcome_message = WELCOME_MESSAGE, -- set to "" if you don't like the fancy godot robot
  --   loading_text = "loading",
  --   question_sign = "", -- you can use emoji if you want e.g. 🙂
  --   answer_sign = "ﮧ", -- 🤖
  --   max_line_length = 120,
  --   yank_register = "+",
  --   chat_layout = {
  --     relative = "editor",
  --     position = "50%",
  --     size = {
  --       height = "80%",
  --       width = "80%",
  --     },
  --   },
  --   settings_window = {
  --     border = {
  --       style = "rounded",
  --       text = {
  --         top = " Settings ",
  --       },
  --     },
  --   },
  --   chat_window = {
  --     filetype = "chatgpt",
  --     border = {
  --       highlight = "FloatBorder",
  --       style = "rounded",
  --       text = {
  --         top = " ChatGPT ",
  --       },
  --     },
  --   },
  --   chat_input = {
  --     prompt = "  ",
  --     border = {
  --       highlight = "FloatBorder",
  --       style = "rounded",
  --       text = {
  --         top_align = "center",
  --         top = " Prompt ",
  --       },
  --     },
  --   },
  --   openai_params = {
  --     model = "text-davinci-003",
  --     frequency_penalty = 0,
  --     presence_penalty = 0,
  --     max_tokens = 300,
  --     temperature = 0,
  --     top_p = 1,
  --     n = 1,
  --   },
  --   openai_edit_params = {
  --     model = "code-davinci-edit-001",
  --     temperature = 0,
  --     top_p = 1,
  --     n = 1,
  --   },
  -- keymaps = {
  --   close = { "<C-c>", "<Esc>" },
  --   yank_last = "<C-y>",
  --   scroll_up = "<C-k>",
  --   scroll_down = "<C-j>",
  --   toggle_settings = "<C-o>",
  --   new_session = "<C-n>",
  --   cycle_windows = "<C-l>",
  -- },
}

local opts = { noremap = true, silent = true }

local keymap = vim.keymap.set

keymap("n", "<c-p>", "<cmd>ChatGPT<cr>", opts)

-- <C-c> to close chat window.
-- <C-u> scroll up chat window.
-- <C-d> scroll down chat window.
-- <C-y> to copy/yank last answer.
-- <C-o> Toggle settings window.
-- <C-n> Start new session.
-- <Tab> Cycle over windows.
-- <C-i> [Edit Window] use response as input.
