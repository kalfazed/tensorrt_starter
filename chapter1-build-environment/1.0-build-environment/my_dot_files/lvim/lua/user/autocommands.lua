-- Autocommands (https://neovim.io/doc/user/autocmd.html)
-- vim.api.nvim_create_autocmd("BufEnter", {
--   pattern = { "*.json", "*.jsonc" },
--   -- enable wrap mode for json files only
--   command = "setlocal wrap",
-- })
-- vim.api.nvim_create_autocmd("FileType", {
--   pattern = "zsh",
--   callback = function()
--     -- let treesitter use bash highlight for zsh files as well
--     require("nvim-treesitter.highlight").attach(0, "bash")
--   end,
-- })

lvim.autocommands = {
  -- .. other cmd
  {
    {"ColorScheme"},
    {
      desc = "Transparent NvimTree on unfocused",
      command = "hi NvimTreeNormalNC guibg=NONE"
    }
  },
  {
    {"BufEnter", "BufWinEnter" },
    {
        group = "lvim_user",
        pattern = {"*.cpp", "*.hpp", "*.c", "*.h", "*.cu"},
        command = "setlocal ts=4 sw=4",
    }
  }
}
