from src.mapping import SchemaMappingEngine

e = SchemaMappingEngine()
for name, cols in {
    "A": ["user_id", "display_name", "email", "phone", "city", "username"],
    "B": ["handle", "name", "email_address", "mobile", "location", "screen_name"],
    "C": ["social_handle", "profile_name", "contact_email", "contact_no", "home_city", "last_active"],
}.items():
    print("===", name, "===")
    for col, p in e.suggest_mappings_for_table(cols).items():
        print(
            f"  {col:20} -> {p['suggested_canonical']:12} "
            f"conf={p['confidence']:.2f} ambig={p.get('is_ambiguous')} | {p['reason']}"
        )
    print()
    mapping = {c: p["suggested_canonical"] for c, p in e.suggest_mappings_for_table(cols).items()}
    ok, errs = e.validate_user_mapping(mapping, cols)
    print("  valid=", ok, "errors=", errs)
    print()
